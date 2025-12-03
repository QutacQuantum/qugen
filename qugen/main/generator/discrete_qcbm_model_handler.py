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

import matplotlib.pyplot as plt 
import json
import time
import hashlib
import os
import cma
import warnings
import pickle

from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

from qugen.main.generator.base_model_handler import BaseModelHandler
from qugen.main.generator.quantum_circuits.discrete_generator_pennylane import generate_samples
from qugen.main.data.data_handler import PITNormalizer, MinMaxNormalizer, NoTransformNormalizer
from qugen.main.data.helper import random_angle, kl_divergence

import pennylane as qml

import matplotlib.pyplot as plt
import matplotlib as mpl


jax.config.update("jax_enable_x64", True)
mpl.use("Agg")


class DiscreteQCBMModelHandler(BaseModelHandler):
    """
    Parameters:

    """
    def __init__(self): 
        """Initialize the model handler by defining all attributes which should immediately be available across all methods."""
        super().__init__()
        self.hist_samples = None
        self.device = 'cpu'
        self.n_qubits = None
        self.n_registers = None
        self.circuit_depth = None
        self.weights = None
        self.generator = None
        self.num_params = None
        self.circuit = None
        self.n_epochs = None
        self.sigma = None
        self.batch_size = None
        self.performed_trainings = 0
        self.hot_start_path = None
        self.weights = None
        self.save_artifacts = None
        self.slower_progress_update = None
        self.key = None


    def build(
        self,
        model_name,
        data_set,
        n_qubits=8,
        n_registers=2,
        circuit_depth=1,
        random_seed=2,
        initial_sigma=2,
        circuit_type='copula',
        transformation='pit',
        hot_start_path='',
        save_artifacts=True,
        slower_progress_update=False
    ) -> BaseModelHandler:
        """Build the discrete QCBM model.
        This defines the architecture of the model, including the circuit ansatz, data transformation and whether the artifacts are saved.

        Args: 
            model_name (str): The name which will be used to save the data to disk.
            data_set: The name of the data set which gets is set as part of the model name
            n_qubits (int, optional): Number of qubits. Defaults to 2.
            n_registers (int, optional): Number of dimensions of the data. Defaults to 2.
            circuit_depth (int, optional): Number of repetitions of qml.StronglyEntanglingLayers. Defaults to 1.
            initial_sigma (float, optional): Initial value of sigma used in the CMA optimizer. Defaults to 2.0
            circuit_type (string, optional): name of the circuit anstaz to be used for the QCBM, either "copula" or "standard". Defaults to "copula"
            transformation (str, optional): Type of normalization, either "minmax", "pit", or "none". Defaults to "pit".
            hot_start_path (str, optional): Path to the location of previously trained model parameters in numpy array format. Defaults to '' which implies that the model will be trained starting with random weights.
            save_artifacts (bool, optional): Whether to save the artifacts to disk. Defaults to True.
            slower_progress_update (bool, optional): Controls how often the progress bar is updated. If set to True, update every 10 seconds at most, otherwise use tqdm defaults. Defaults to False.

        Returns:
            BaseModelHandler: Return the built model handler. It is not strictly necessary to overwrite the existing variable with this
            since all changes are made in place.
            """
        self.slower_progress_update = slower_progress_update
        self.save_artifacts = save_artifacts
        self.n_qubits = n_qubits
        self.n_registers = n_registers
        self.circuit_depth = circuit_depth
        self.data_set = data_set
        self.sigma= initial_sigma
        self.hot_start_path = hot_start_path
        time_str = str(time.time()).encode('utf-8')
        uniq = hashlib.md5(time_str).hexdigest()[:4]
        self.transformation = transformation
        self.circuit_type = circuit_type

        self.model_name = model_name + '_'  + self.data_set+ '_' + self.circuit_type + '_' + self.transformation+ '_' + 'qcbm_' + uniq 

        # jax specific
        self.key = jax.random.PRNGKey(random_seed)

        if self.circuit_type == 'copula' and self.transformation not in ['pit']:
            raise ValueError("Copula circuit must have PIT transformation. Current transformation: " + self.transformation)

       
        # create list of minimum and maximum of bins per register for histogram creation 
        all_bins = []
        all_ranges = []
        n = 2 ** (self.n_qubits // self.n_registers)
        for _ in range(self.n_registers):
            all_bins.append(n)
            if self.transformation in ['minmax', 'pit']:
                all_ranges.append([0, 1])  # Normalized data range
            else:
                all_ranges.append(None)  # Will be set dynamically for 'none' transformation
        self.all_bins = all_bins
        self.all_ranges = all_ranges
        self.path_to_models = "experiments/" + self.model_name

        self.metadata = dict({
            'model_name': self.model_name,
            'data_set ': self.data_set,
            'n_qubits': self.n_qubits,
            'n_registers': self.n_registers,
            'circuit_type': self.circuit_type,
            'circuit_depth': self.circuit_depth,
            'transformation': self.transformation,
            'hot_start_path': self.hot_start_path,
            'num_params': None,  # Will be set after generator is created
            'circuit_info': {
                'generator_architecture': f'{self.circuit_type}_circuit',
                'total_qubits': self.n_qubits,
                'qubits_per_register': self.n_qubits // self.n_registers,
            },
            'timestamp': time.time(),
            "training_data": {},
            "performance_metrics": {},
        })


        if save_artifacts:
            os.makedirs(self.path_to_models)
            print('model_name', self.model_name)
            with open(
                self.path_to_models + "/" + "meta.json", "w"
            ) as fp:
                json.dump(self.metadata, fp)

        if self.circuit_type == 'copula':
            from qugen.main.generator.quantum_circuits.discrete_generator_pennylane \
                import discrete_copula_circuit_JAX as get_generator
        elif self.circuit_type == 'standard':
            from qugen.main.generator.quantum_circuits.discrete_generator_pennylane \
                import discrete_standard_circuit_JAX as get_generator
        else:
            raise ValueError("Circuit value must be either 'standard' or 'copula'")
        self.generator, self.num_params = get_generator(self.n_qubits, self.n_registers, self.circuit_depth)
        
        # Update metadata with actual parameter count
        self.metadata['num_params'] = self.num_params
        self.metadata['circuit_info']['generator_params'] = self.num_params
        
        return self


    def save(self, file_path: Path, overwrite: bool = True) -> BaseModelHandler:
        """Save the generator weights to disk.

        Args:
            file_path (Path): The paths of the pickled generator weights.
            overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to True.

        Returns:
            BaseModelHandler: The model, unchanged.
        """
        if overwrite or not os.path.exists(file_path):
            with open(file_path, "wb") as file:
                pickle.dump(self.generator_weights, file)
        return self

    def reload(
        self, model_name: str, epoch: int, random_seed: Optional[int] = None
    ) -> BaseModelHandler:
        """Reload the model parameters and the lastest sigma for the continuing training of the generator from the file weights_file.

        Args:
            weights_file (str): The path to the pickled tuple containing the generator weights and sigma value.

        Returns:
            BaseModelHandler: The model, but changes have been made in place as well.
        """

        weights_file = "experiments/" + model_name + "/" + "parameters_training_iteration={0}.npy".format(str(epoch))
        meta_file = "experiments/"+ model_name + "/" +  "meta.json"
        reverse_file = "experiments/" + model_name + "/" + 'reverse_lookup.npy'

        self.weights, self.sigma = np.load(weights_file, allow_pickle=True)
        self.reverse_lookup = jnp.load(reverse_file)
        
        with open(meta_file, 'r') as f:
            self.metadata = json.load(f)

        if random_seed is None:
            if self.key is None:
                self.key = jax.random.PRNGKey(2)
        else:
            self.key = jax.random.PRNGKey(random_seed)
            if self.key is not None:
                warnings.warn(
                    "Random state already initalized in the model handler, but a random_seed was specified when reloading."
                )

        self.n_qubits = self.metadata["n_qubits"]
        self.n_registers = self.metadata["n_registers"]
        self.circuit_depth = self.metadata["circuit_depth"]
        self.transformation = self.metadata["transformation"]
        self.performed_trainings = len(self.metadata["training_data"])
        self.circuit_type = self.metadata['circuit_type']

        if self.generator is None:
            if self.circuit_type == 'copula':
                from qugen.main.generator.quantum_circuits.discrete_generator_pennylane \
                    import discrete_copula_circuit_JAX as get_generator
            elif self.circuit_type == 'standard':
                from qugen.main.generator.quantum_circuits.discrete_generator_pennylane \
                    import discrete_standard_circuit_JAX as get_generator
            else:
                raise ValueError("Circuit value must be either 'standard' or 'copula'")
            self.generator, self.num_params = get_generator(self.n_qubits, self.n_registers, self.circuit_depth)
        return self

    def load(
        self,
        model_name_or_path: str,
        epoch: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> 'DiscreteQCBMModelHandler':
        """Load a trained DiscreteQCBM model from a specified directory.

        This method initializes the current instance with all saved model details
        from meta.json, automatically loading the best epoch if none is specified.

        Args:
            model_name_or_path (str): Either:
                - A model directory name (will look in experiments/ directory)
                - An absolute path to a model directory
                - A relative path to a model directory
            epoch (int, optional): Specific epoch to load. If None, loads the best epoch
                                  based on lowest KL divergence in performance_metrics.
            random_seed (int, optional): Random seed for reproducibility.

        Returns:
            DiscreteQCBMModelHandler: The current instance, now initialized with loaded model.

        Raises:
            FileNotFoundError: If the model directory or required files don't exist.
            ValueError: If the model configuration is invalid.
        """

        # Determine if this is a path or just a model name
        if os.path.exists(model_name_or_path):
            # It's a valid path to an existing directory
            self.path_to_models = model_name_or_path
            self.model_name = os.path.basename(model_name_or_path)
        elif '/' in model_name_or_path or '\\' in model_name_or_path:
            # It looks like a path but doesn't exist yet - try to use it anyway
            # (will fail with clear error message below if directory doesn't exist)
            self.path_to_models = model_name_or_path
            self.model_name = os.path.basename(model_name_or_path)
        else:
            # It's just a model name, use default experiments directory
            self.model_name = model_name_or_path
            self.path_to_models = f"experiments/{model_name_or_path}"
        meta_file = f"{self.path_to_models}/meta.json"
        reverse_file = f"{self.path_to_models}/reverse_lookup.npy"

        # Verify model directory exists
        if not os.path.exists(self.path_to_models):
            raise FileNotFoundError(f"Model directory not found: {self.path_to_models}")

        # Load metadata
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        with open(meta_file, 'r') as f:
            self.metadata = json.load(f)

        # Extract core model configuration from metadata
        self.n_qubits = self.metadata["n_qubits"]
        self.n_registers = self.metadata["n_registers"]
        self.circuit_depth = self.metadata["circuit_depth"]
        self.transformation = self.metadata["transformation"]
        self.circuit_type = self.metadata["circuit_type"]
        self.data_set = self.metadata.get("data_set ", "").strip()  # Note: typo in original
        self.hot_start_path = self.metadata.get("hot_start_path", "")
        self.performed_trainings = len(self.metadata.get("training_data", {}))

        # Determine which epoch to load
        if epoch is None:
            # Find best epoch based on lowest KL divergence
            performance_metrics = self.metadata.get("performance_metrics", {})
            if not performance_metrics:
                raise ValueError(f"No performance metrics found in {meta_file}. Cannot auto-select best epoch.")

            best_epoch = None
            best_kl = float('inf')

            for epoch_key, metrics in performance_metrics.items():
                if epoch_key.startswith('epoch_'):
                    epoch_num = int(epoch_key.replace('epoch_', ''))
                    kl_div = metrics.get('kl_divergence_transformed', float('inf'))
                    if kl_div < best_kl:
                        best_kl = kl_div
                        best_epoch = epoch_num

            if best_epoch is None:
                raise ValueError("Could not determine best epoch from performance metrics.")

            epoch = best_epoch
            print(f"Auto-selected best epoch {epoch} with KL divergence: {best_kl:.6f}")

        # Load weights and sigma for the specified epoch
        weights_file = f"{self.path_to_models}/parameters_training_iteration={epoch}.npy"
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"Weights file not found: {weights_file}")

        self.weights, self.sigma = np.load(weights_file, allow_pickle=True)

        # Load reverse lookup table
        if os.path.exists(reverse_file):
            self.reverse_lookup = jnp.load(reverse_file)
        else:
            print(f"Warning: Reverse lookup file not found: {reverse_file}")
            self.reverse_lookup = None

        # Set up random key
        if random_seed is None:
            self.key = jax.random.PRNGKey(2)  # Default seed
        else:
            self.key = jax.random.PRNGKey(random_seed)

        # Create quantum circuit generator
        if self.circuit_type == 'copula':
            from qugen.main.generator.quantum_circuits.discrete_generator_pennylane \
                import discrete_copula_circuit_JAX as get_generator
        elif self.circuit_type == 'standard':
            from qugen.main.generator.quantum_circuits.discrete_generator_pennylane \
                import discrete_standard_circuit_JAX as get_generator
        else:
            raise ValueError(f"Invalid circuit type: {self.circuit_type}. Must be 'copula' or 'standard'.")

        self.generator, self.num_params = get_generator(
            self.n_qubits, self.n_registers, self.circuit_depth
        )

        # Set up histogram bins and ranges for prediction
        n = 2 ** (self.n_qubits // self.n_registers)
        self.all_bins = [n] * self.n_registers

        if self.transformation in ['minmax', 'pit']:
            self.all_ranges = [[0, 1]] * self.n_registers
        else:  # transformation == 'none'
            self.all_ranges = [None] * self.n_registers

        # Other necessary attributes
        self.save_artifacts = True  # Default for loaded models
        self.slower_progress_update = False
        self.hist_samples = None
        self.batch_size = None
        self.n_epochs = None
        self.device = 'cpu'

        print(f"Successfully loaded model '{self.model_name}' from epoch {epoch}")
        print(f"Model configuration: {self.n_qubits} qubits, {self.n_registers} registers, "
              f"circuit_type='{self.circuit_type}', transformation='{self.transformation}'")

        return self

    def load_best(self, model_name_or_path: str, random_seed: Optional[int] = None) -> 'DiscreteQCBMModelHandler':
        """Convenience method to load the best performing epoch of a trained model.

        Args:
            model_name_or_path (str): Either:
                - A model directory name (will look in experiments/ directory)
                - An absolute path to a model directory
                - A relative path to a model directory
            random_seed (int, optional): Random seed for reproducibility.

        Returns:
            DiscreteQCBMModelHandler: The current instance, now initialized with the best model.
        """
        return self.load(model_name_or_path=model_name_or_path, epoch=None, random_seed=random_seed)


    def plot_training_data(self, train_dataset: np.array):
        """ Plot training data and compute an estimate of the true probability distribution """
        size = (5, 5)
        plt.rcParams["figure.figsize"] = size

        data_histogram = np.histogramdd(train_dataset, bins=self.all_bins, range=self.all_ranges)
        true_probability = data_histogram[0]/np.sum(data_histogram[0])

        if self.n_registers ==2:
            plt.imshow((true_probability), interpolation='none')
        elif self.n_registers ==3:   
            z_slice = 1
            plt.imshow(true_probability[:, :, z_slice], interpolation='none')
        plt.show()  


    def evaluator(self, solutions):
        """Computes the loss function for all candidate solutions from CMA

        Args:
            solutions (list): List of the potential weights which the CMA algorithm has sampled.

        Returns:
            loss (list): List of all training losses corresponding to each entry in solutions.
        """        
        
        self.key, subkey = jax.random.split(self.key)

        v_generator = jax.vmap(
            lambda weights: self.generator(
            subkey,
            weights,
            n_shots=self.hist_samples,
            )
        )

        solutions = jnp.array(solutions)
        binary_samples = v_generator(solutions)

        def iterate_samples(subkey, binary_sample):
            return generate_samples(subkey, binary_sample, self.n_registers, self.n_qubits, noisy=False)

        keys = jax.random.split(self.key, num=binary_samples.shape[0] + 1)
        self.key, subkeys = keys[0], keys[1:]
        samples = jax.vmap(iterate_samples)(subkeys, binary_samples)

        def iterate_loss(sample):
            learned_histogram = jnp.histogramdd(sample, bins=self.all_bins, range=self.all_ranges)
            learned_probability = learned_histogram[0]/jnp.sum(learned_histogram[0])
            return kl_divergence(self.true_probability, learned_probability)
        loss = jax.vmap(iterate_loss)(samples)
        return loss.tolist()


    def train(
        self,
        train_dataset: np.array,
        n_epochs = 500, 
        batch_size = 200, 
        hist_samples = 10000, 
        plot_training_data = False, 
        
    ) -> BaseModelHandler:

        """Train the discrete QCBM.

        Args:
            train_dataset (np.array): The training dataset.
            n_epochs (int): The number of epochs.
            batch_size (int, optional): The population size used for the CMA optimizer. Defaults to 200
            hist_samples (int, optional): Number of samples generated from the generator at every epoch to compute the loss fucntion. Defaults to 1e4
            plot_training_data (bool): If True, a plot of the training data is displayed for debugging purposes     

        Returns:
            BaseModelHandler: The trained model.
        """

        self.n_epochs = n_epochs # less data, so we need more epochs
        self.batch_size = batch_size #aka population size 
        self.hist_samples = hist_samples

        if self.transformation == 'minmax':
            self.normalizer = MinMaxNormalizer(epsilon=1e-6)
        elif self.transformation == 'pit':
            self.normalizer = PITNormalizer(epsilon=1e-6) 
        elif self.transformation == 'none':
            self.normalizer = NoTransformNormalizer(epsilon=1e-6)
        else:
            raise ValueError("Transformation value must be either 'minmax', 'pit', or 'none'")    

        train_dataset = self.normalizer.fit_transform(train_dataset)
        self.reverse_lookup = self.normalizer.reverse_lookup
        
        # Update histogram ranges for 'none' transformation based on actual data
        if self.transformation == 'none':
            data_min = train_dataset.min(axis=0)
            data_max = train_dataset.max(axis=0)
            for i in range(self.n_registers):
                self.all_ranges[i] = [data_min[i], data_max[i]]

        if self.performed_trainings == 0:
            self.previous_trained_epochs = 0 
        else:
            self.previous_trained_epochs = sum([self.metadata["training_data"][str(i)]["n_epochs"] for i in range(self.performed_trainings)])        

        training_data = {}
        training_data["batch_size"] = self.batch_size
        training_data["n_epochs"] = self.n_epochs
        training_data["sigma"] = self.sigma
        self.metadata["training_data"][str(self.performed_trainings)] = training_data
        self.performed_trainings += 1
        if self.save_artifacts:
            with open(self.path_to_models + "/" + "meta.json", "w+") as file:
                    json.dump(self.metadata, file)

            jnp.save(self.path_to_models + "/" + 'reverse_lookup.npy', self.reverse_lookup)

        self.data_histogram = np.histogramdd(train_dataset, bins=self.all_bins, range=self.all_ranges)
        self.true_probability = self.data_histogram[0]/np.sum(self.data_histogram[0])

        if plot_training_data ==True:
            self.plot_training_data(train_dataset)

        # Try to upload pre-trained parameters for higher depth and default to random angle. 
        if self.weights is not None: 
            x0 = self.weights
            print('Training starting from lastest model parameters')

        elif self.hot_start_path == '':
            x0 = random_angle(self.num_params)
            print('Training starting from random parameter values')
        else:
            try:
                init_params, _ = np.load(self.hot_start_path, allow_pickle=True)
                x0 = np.zeros(self.num_params)
                x0[:len(init_params)] = init_params
                print(f'Training starting from parameters in path {self.hot_start_path}')
            except FileNotFoundError:
                warnings.warn("Cannot find hot start parameters file, defaulting to using random parameter values")
                x0 = random_angle(self.num_params)

        iter = 0
        print(f'starting training with sigma at value {self.sigma}')       
        log = [('iteration', 'n_epochs', 'training_batch_ratio', 'kl_div_transformed', 'time')]
        bounds = [self.num_params * [-np.pi], self.num_params * [np.pi]]
        options = {'bounds': bounds, 'maxfevals': self.n_epochs*self.batch_size, 'popsize': self.batch_size, 'verbose': -3}
        es = cma.CMAEvolutionStrategy(x0, self.sigma, options)
        mininterval = 10
        time_of_last_update = time.time()
        while not es.stop():
            t_0 = time.time()
            solutions = es.ask()
            loss = self.evaluator(solutions)
            es.tell(solutions, loss)
            iter += 1
            if self.slower_progress_update:
                cand_time = time.time()
                time_since_last_update = cand_time - time_of_last_update
                if time_since_last_update >= mininterval:
                    es.disp()
                    time_of_last_update = cand_time

            else:
                es.disp()
            log.append((iter, self.n_epochs, len(train_dataset) // self.batch_size, es.result[1],
                         time.time() - t_0))
            # save the logged process and the current weights to file
            self.weights = es.result[0]
            last_sigma = es.sigma
            # Save artifacts every 100 iterations to reduce I/O
            if self.save_artifacts and iter % 100 == 0:
                current_epoch = iter + self.previous_trained_epochs
                file_path = f"{self.path_to_models}/parameters_training_iteration={current_epoch}"
                np.save(file_path, np.array([self.weights, last_sigma], dtype=object))
                np.save(self.path_to_models+ '/log_' + str(current_epoch), np.array(log))
                
                # Update performance metrics in metadata
                self.metadata['performance_metrics'][f'epoch_{current_epoch}'] = {
                    'kl_divergence_transformed': float(es.result[1]),
                    'sigma': float(last_sigma),
                    'iteration': current_epoch,
                }
                
                # Save QASM representation every 100 iterations if artifacts are saved
                try:
                    self.save_circuit_qasm(epoch=current_epoch)
                except Exception as e:
                    print(f"Warning: Could not save QASM for epoch {current_epoch}: {e}")
                
                # Update metadata file
                with open(self.path_to_models + "/" + "meta.json", "w+") as file:
                    json.dump(self.metadata, file)

            t_0 = time.time()
        self.sigma = last_sigma
        return self
        
     
    def predict(self,
        n_samples: int,
    ) -> np.array:
        """Generate samples from the trained model and perform the inverse of the data transformation
        which was used to transform the training data to be able to compute the KL-divergence in the original space.

        Args:
            n_samples (int, optional): Number of samples to generate.

        Returns:
            np.array: Array of samples of shape (n_samples, sample_dimension).
        """        
        samples_transformed = self.predict_transform(n_samples)

        if self.transformation == 'pit':
            self.transformer = PITNormalizer(epsilon=1e-6)
        elif self.transformation == 'minmax':  
            self.transformer = MinMaxNormalizer(epsilon=1e-6)
        elif self.transformation == 'none':
            self.transformer = NoTransformNormalizer(epsilon=1e-6)
            
        self.transformer.reverse_lookup = self.reverse_lookup
        samples = self.transformer.inverse_transform(samples_transformed)
        return samples

    def predict_transform(self,
        n_samples: int,
    ) -> np.array:
        """Generate samples from the trained model in the transformed space (the n-dimensional unit cube).

        Args:
            n_samples (int, optional): Number of samples to generate.

        Returns:
            np.array: Array of samples of shape (n_samples, sample_dimension).
        """     
        if self.performed_trainings == 0:
            raise ValueError(
                    "Please train the model before trying to generate samples."
    )
        
        # JAX
        self.key, subkey = jax.random.split(self.key)
        binary_samples = self.generator(
                subkey,
                weights=jnp.array(self.weights),
                n_shots=n_samples,
        )
        self.key, subkey = jax.random.split(self.key)
        samples = generate_samples(subkey, binary_samples, self.n_registers, self.n_qubits, noisy=True)

        samples_transformed = np.array(samples)        
        
        return samples_transformed    

    def get_circuit_qasm_string(self) -> str:
        """Generate QASM string representation of the quantum circuit with current weights.

        Relies on pennylane-qiskit plugin for QASM export.

        Critical Bug: Does not handle issingXX gates correctly
        
        Returns:
            str: QASM string representation of the quantum circuit.
        """
        if self.weights is None:
            raise ValueError("Circuit weights not available. Train the model first or reload weights.")
        
        # Create a temporary device for circuit visualization
        dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Create a dummy circuit to visualize the structure
        @qml.qnode(dev)
        def circuit_for_qasm():
            # Create a simplified version for visualization
            if self.circuit_type == 'copula':
                self._build_copula_circuit_for_qasm()
            else:
                self._build_standard_circuit_for_qasm()
            
            return qml.sample(wires=range(self.n_qubits))
        
        # Get the actual QASM representation
        circuit_text = qml.to_openqasm(circuit_for_qasm)()
        
        # Add header with circuit information
        qasm_header = f"""// Quantum Circuit from QuGen DiscreteQCBMModelHandler
// Model: {self.model_name}
// Qubits: {self.n_qubits}
// Registers: {self.n_registers}
// Circuit Type: {self.circuit_type}
// Circuit Depth: {self.circuit_depth}
// Parameters: {self.num_params}
// Output: Samples probability distribution across all qubit states

"""
        
        return qasm_header + circuit_text

    def _build_copula_circuit_for_qasm(self):
        """Build the actual copula circuit structure for QASM visualization."""
        from itertools import combinations
        
        n = self.n_qubits // self.n_registers
        
        # 1. Copula block (Hadamards + inter-register CNOTs)
        for i in range(n):
            qml.Hadamard(wires=i)
        
        for j in range(self.n_registers - 1):
            for k in range(n):
                qml.CNOT(wires=[k, k + n * (j + 1)])
        
        # 2. Parametric part (matching copula_parametric from discrete_generator_pennylane.py)
        param_idx = 0
        for _ in range(self.circuit_depth):
            # RZ-RX-RZ rotations on all qubits
            for k in range(n):
                for j in range(self.n_registers):
                    wire = j * n + k
                    if param_idx < len(self.weights):
                        qml.RZ(self.weights[param_idx], wires=wire)
                        param_idx += 1
                    if param_idx < len(self.weights):
                        qml.RX(self.weights[param_idx], wires=wire)
                        param_idx += 1
                    if param_idx < len(self.weights):
                        qml.RZ(self.weights[param_idx], wires=wire)
                        param_idx += 1
            
            # IsingXX gates for entanglement within each register
            for i, j in combinations(range(n), 2):
                for l in range(self.n_registers):
                    if param_idx < len(self.weights):
                        qml.IsingXX(self.weights[param_idx], wires=[l * n + i, l * n + j])
                        param_idx += 1

    def _build_standard_circuit_for_qasm(self):
        """Build the actual standard circuit structure for QASM visualization."""
        param_idx = 0
        
        for _ in range(self.circuit_depth):
            # 1. RY rotations on all qubits
            for k in range(self.n_qubits):
                if param_idx < len(self.weights):
                    qml.RY(self.weights[param_idx], wires=k)
                    param_idx += 1
            
            # 2. IsingYY gates between adjacent qubits
            for k in range(self.n_qubits - 1):
                qubit_1 = k
                qubit_2 = k + 1
                if param_idx < len(self.weights):
                    qml.IsingYY(self.weights[param_idx], wires=[qubit_1, qubit_2])
                    param_idx += 1
            
            # 3. CRY (Controlled RY) gates between adjacent qubits
            for k in range(self.n_qubits - 1):
                control_qubit = k
                target_qubit = k + 1
                if param_idx < len(self.weights):
                    qml.CRY(self.weights[param_idx], wires=[control_qubit, target_qubit])
                    param_idx += 1

    def save_circuit_qasm(self, file_path: str = None, epoch: int = None) -> str:
        """Save the quantum circuit as a QASM string to file.
        
        Args:
            file_path (str, optional): Custom file path. If None, uses default naming.
            epoch (int, optional): Training epoch number for filename.
            
        Returns:
            str: Path to the saved QASM file.
        """
        if file_path is None:
            if epoch is not None:
                file_path = f"{self.path_to_models}/circuit_epoch_{epoch}.qasm"
            else:
                file_path = f"{self.path_to_models}/circuit_current.qasm"
        
        qasm_string = self.get_circuit_qasm_string()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(qasm_string)
        
        return file_path

