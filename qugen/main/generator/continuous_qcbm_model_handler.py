# Copyright 2023 QUTAC, BASF Digital Solutions GmbH, BMW Group, 
# Lufthansa Industry Solutions AS GmbH, Merck KGaA (Darmstadt, Germany), 
# Munich Re, SAP SE.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from tqdm import tqdm

import json
import time
import hashlib
import os
import cma

import pickle
import jax
import jax.numpy as jnp
import numpy as np

from jax.config import config
config.update("jax_enable_x64", False)

from qugen.main.generator.base_model_handler import BaseModelHandler
from qugen.main.data.data_handler import PITNormalizer, MinMaxNormalizer
from qugen.main.data.helper import kl_divergence

from qugen.main.generator.quantum_circuits.continuous_circuits import get_qnode


class ContinuousQCBMModelHandler(BaseModelHandler):
    """
    Parameters:

    """
    def __init__(self): 
        """Initialize the model handler by defining all attributes which should immediately be available across all methods."""
        super().__init__()
        self.n_qubits = None
        self.circuit_depth = None
        self.weights = None
        self.num_params = None
        self.performed_trainings = 0
        self.circuit = None
        self.generator = None
        self.sigma = None
        self.save_artifacts = None
        self.slower_progress_update = None
        self.normalizer = None

    def build(
        self,
        model_name: str,
        data_set: str,
        n_qubits: int,
        circuit_depth: int,
        random_seed: int = 42,
        transformation: str = "pit",
        initial_sigma: float = 1e-2,
        save_artifacts=True,
        slower_progress_update=False
    ) -> BaseModelHandler:
        """Build the discrete qcbm model.

        Args:
            model_name (str): The name which will be used to save the data to disk.
            data_set (str): The name of the data set which gets as part of the model name
            n_qubits (int, optional): Number of qubits. Defaults to 2.
            circuit_depth (int, optional): Number of repetitions of qml.StronglyEntanglingLayers. Defaults to 1.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
            transformation (str, optional): Type of normalization, either "minmax" or "pit". Defaults to "pit".
            initial_sigma (float): Initial value of the CMA optimization parameter
            save_artifacts (bool, optional): Whether to save the model artifacts to disk. Defaults to True.
            slower_progress_update (bool, optional): Whether to update the progress bar less frequently. Defaults to False.

        Returns:
            BaseModelHandler: The built model.
         """
        self.slower_progress_update = slower_progress_update
        self.save_artifacts = save_artifacts
        self.random_key = jax.random.PRNGKey(random_seed)
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.transformation = transformation 
        self.n_samples = 10000
        self.sigma = initial_sigma

        time_str = str(time.time()).encode("utf-8")
        uniq = hashlib.md5(time_str).hexdigest()[:4]
        self.data_set = data_set
        self.circuit_type = "continuous"
        self.model_name = model_name + '_'  + self.data_set + '_' + self.transformation + '_' + 'qcbm_' + uniq 
        self.path_to_models = "experiments/" + self.model_name
        print(self.model_name)
        self.metadata = dict(
            {
                "model_name": self.model_name,
                "data_set":  self.data_set,
                "n_qubits": self.n_qubits,
                "circuit_type": self.circuit_type,
                "circuit_depth": self.circuit_depth,
                "transformation": self.transformation,
                "training_data": {},
            }
        )
        if save_artifacts:
            os.makedirs(self.path_to_models)
            with open(
                self.path_to_models + "/" + "meta.json", "w"
            ) as fp:
                json.dump(self.metadata, fp)
        if self.transformation == 'minmax':
            self.normalizer = MinMaxNormalizer()
        elif self.transformation == 'pit':
            self.normalizer = PITNormalizer()
        else:
            raise ValueError("Transformation value must be either 'minmax' or 'pit'")
        self.generator, self.num_params = get_qnode(self.circuit_depth, self.n_qubits)
        return self


    def save(self, file_path: Path, overwrite: bool = True) -> BaseModelHandler:
        """Save the generator weights to disk.

        Args:
            file_path (Path): The paths where the pickled tuple of generator and discriminator weights will be placed.
            overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to True.

        Returns:
            BaseModelHandler: The model, unchanged.
        """
        if overwrite or not os.path.exists(file_path):
            with open(file_path, "wb") as file:
                pickle.dump((self.weights), file)
        return self


    def reload(self, model_name: str, epoch: int) -> BaseModelHandler:
        """Reload the parameters for the generator and the discriminator from the file weights_file.

        Args:
            weights_file (str): The path to the pickled tuple containing the generator and discriminator weights.

        Returns:
            BaseModelHandler: The model, but changes have been made in place as well.
        """
        weights_file = "experiments/" + model_name + "/" + "parameters_training_iteration={0}.npy".format(str(epoch))
        meta_file = "experiments/"+ model_name + "/" +  "meta.json"
        reverse_file = "experiments/" + model_name + "/" + 'reverse_lookup.npy'

        self.model_name = model_name
        self.path_to_models = "experiments/" + self.model_name
        self.weights, self.sigma = np.load(weights_file, allow_pickle=True)
        with open(meta_file, "r") as file:
            self.metadata = json.load(file)

        self.reverse_lookup = jnp.load(reverse_file)
        self.n_qubits = self.metadata["n_qubits"]
        self.transformation = self.metadata["transformation"]
        self.circuit_depth = self.metadata["circuit_depth"]
        self.performed_trainings = len(self.metadata["training_data"])
        self.random_key = jax.random.PRNGKey(2)

        if self.normalizer is None:
            if self.transformation == 'minmax':
                self.normalizer = MinMaxNormalizer()
            elif self.transformation == 'pit':
                self.normalizer = PITNormalizer()
            else:
                raise ValueError("Transformation value must be either 'minmax' or 'pit'")
        self.normalizer.reverse_lookup = self.reverse_lookup
        if self.generator is None:
            self.generator, self.num_params = get_qnode(self.circuit_depth, self.n_qubits)
        return self

    def cost(self, weights, noise):
        res = self.v_qnode(noise, weights)
        res = (jnp.array(res)+1)/2

        bins = [16 for _ in range(self.n_qubits)]
        bin_range = [(0, 1) for _ in range(self.n_qubits)]
        histogram_samples = jnp.histogramdd(res, bins=bins, range=bin_range) #, requires_grad=True)
        probability_samples = histogram_samples[0]/np.sum( histogram_samples[0])

        return kl_divergence(self.original_probability_samples, probability_samples)

    
    def evaluator(self, solutions, noise): 
        jnp_weights = jnp.array([jnp.array(np.reshape(solution, self.weights_shape)) for solution in solutions])
        return self.v_cost(jnp_weights, noise).tolist()


    def train(
        self,
        train_dataset: np.array,
        n_epochs: int = 500, 
        batch_size: int = 200, 
        hist_samples: int = 10000
    ) -> BaseModelHandler:

        self.n_epochs = n_epochs
        self.batch_size = batch_size # aka population size 
        self.hist_samples = hist_samples

        train = self.normalizer.fit_transform(train_dataset)
        self.reverse_lookup = self.normalizer.reverse_lookup


        if self.performed_trainings == 0:
            self.previous_trained_epochs = 0 
        else:
            self.previous_trained_epochs = sum([self.metadata["training_data"][str(i)]["n_epochs"] for i in range(self.performed_trainings)])        

        training_data = {}
        training_data["batch_size"] = self.batch_size
        training_data["n_epochs"] = self.n_epochs
        training_data["sigma"] = self.sigma
        self.metadata["training_data"][str(self.performed_trainings)] = training_data

        if self.save_artifacts:
            with open(self.path_to_models + "/" + "meta.json", "w+") as file:
                json.dump(self.metadata, file)

            jnp.save(self.path_to_models + "/" + 'reverse_lookup.npy', self.reverse_lookup)

        bins = [16 for _ in range(self.n_qubits)]
        bin_range = [(0, 1) for _ in range(self.n_qubits)]
        histogram_samples = np.histogramdd(train, bins=bins, range=bin_range)
        self.original_probability_samples = histogram_samples[0]/np.sum(histogram_samples[0])

        # Try to upload pre-trained parameters for higher depth and default to random angle. 
        if self.weights is not None: 
            x0 = self.weights
            print('Training starting from lastest model parameters')
        else:
            self.random_key, subkey = jax.random.split(self.random_key)
            self.weights = jax.random.uniform(subkey, shape=(self.circuit_depth, 1, self.n_qubits, 3)) * 2*jnp.pi - jnp.pi
            print('Training starting from random parameter values')
        self.weights_shape = self.weights.shape
        self.num_params = self.weights.size
        if self.circuit_depth == 1:
            x0 = np.reshape(self.weights, (self.num_params,))
        elif self.circuit_depth >= 2:
            x0 = self.weights.flatten()
        
        # Optimization with CMA-ES

        iter = 0
        KL = []
        bounds = [self.num_params * [-np.pi], self.num_params * [np.pi]]
        options = {'maxiter': self.n_epochs*self.batch_size,'bounds': bounds, 'maxfevals': self.n_epochs*self.batch_size, 'popsize': self.batch_size, 'verbose': -3}
        es = cma.CMAEvolutionStrategy(x0, self.sigma, options)

        print('maxfevals', self.n_epochs*self.batch_size)

        best_parameters = None
        best_observed_cost = 1000000000
        self.random_key, subkey = jax.random.split(self.random_key)
        noise = jax.random.uniform(subkey, shape=(self.n_samples, self.n_qubits))*2*jnp.pi - jnp.pi
        noise = jnp.array(noise)
        # + 1 because CMA always does one extra iteration, meaning that it stops after 1200 fevals even if maxevals is
        # 1000 with batch size 200
        mininterval = 10
        iterator = tqdm(
            range((self.n_epochs + 1) * self.batch_size), mininterval=mininterval
        )
        pbar_advancement_since_last_update = 0
        time_of_last_update = time.time()
        self.v_qnode = jax.vmap(self.generator, in_axes=(0, None))
        self.v_cost = jax.vmap(self.cost, in_axes=(0, None))

        with iterator as pbar:
            while not es.stop():
                solutions = es.ask()
                loss = self.evaluator(solutions, noise)
                es.tell(solutions, loss)

                KL.append(es.result[1])
                iter += 1
                # See https://github.com/CMA-ES/pycma/blob/development/cma/evolution_strategy.py
                # The CMAEvolutionStrategy.disp() method provides the terminal output, and its formatting exactly
                # corresponds to the values in the directory below.
                terminal_output = {
                    "function_value": "%.15e" % (min(es.fit.fit)),
                    "sigma": "%6.2e" % es.sigma,
                }
                pbar.set_postfix(terminal_output, refresh=False)
                if self.slower_progress_update:
                    cand_time = time.time()
                    time_since_last_update = cand_time - time_of_last_update
                    pbar_advancement_since_last_update += self.batch_size
                    if time_since_last_update >= mininterval:
                        pbar.update(pbar_advancement_since_last_update)
                        time_of_last_update = cand_time
                        pbar_advancement_since_last_update = 0

                else:
                    pbar.update(self.batch_size)
                if es.result[1] < best_observed_cost:
                    best_parameters = es.result[0]
                    best_observed_cost = es.result[1]
                last_sigma = es.sigma
                self.weights = jnp.array(np.reshape(best_parameters, self.weights_shape))

                if self.save_artifacts:
                    file_path = f"{self.path_to_models}/parameters_training_iteration={iter + self.previous_trained_epochs}"
                    np.save(file_path, np.array([self.weights, last_sigma], dtype=object))
  
        self.random_key, subkey = jax.random.split(self.random_key)
        self.sigma = last_sigma
        noise = jax.random.uniform(subkey, shape=(self.n_samples, self.n_qubits)) *2*np.pi - np.pi
        v_qnode = jax.vmap(lambda inpt: self.generator(inpt, self.weights))
        res = v_qnode(noise)
        res = (np.array(res)+1)/2
        self.performed_trainings += 1


    def predict(
            self,
            n_samples: int = 32,
        ) -> np.array:
            """Generate samples from the trained model in the original space.

            Args:
                n_samples (int, optional): Number of samples to generate. Defaults to 32.

            Returns:
                np.array: Array of samples of shape (n_samples, sample_dimension).
            """
            samples_transformed = self.predict_transform(n_samples)
        
            if self.transformation == 'pit':
                self.transformer = PITNormalizer()
            elif self.transformation == 'minmax':  
                self.transformer = MinMaxNormalizer()        
            self.transformer.reverse_lookup = self.reverse_lookup
            samples = self.transformer.inverse_transform(samples_transformed)
            return samples

        
    def predict_transform(self,
        n_samples: int = 32,
    ) -> np.array:
        """Generate samples from the trained model in the transformed space.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 32.

        Returns:
            np.array: Array of samples of shape (n_samples, sample_dimension).
        """
        if self.performed_trainings == 0:
            raise ValueError(
                "Please train the model before trying to generate samples."
            )
        self.random_key, subkey = jax.random.split(self.random_key)
        noise = jax.random.uniform(subkey, (n_samples, self.n_qubits))*2*jnp.pi - jnp.pi
        v_qnode = jax.vmap(lambda inpt: self.generator(inpt, self.weights))
        samples_transformed = v_qnode(noise)
        samples_transformed = (np.asarray(samples_transformed) + 1) / 2

        return samples_transformed  

    def sample(self, n_samples: int = 32):
        """Generate samples from the trained model.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 32.

        Returns:
            np.array: Array of samples of shape (n_samples, sample_dimension).
        """
        return self.predict(n_samples)
