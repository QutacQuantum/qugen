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


import json
import time
import hashlib
import os
import pickle
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import numpy as np

from jax.config import config
config.update("jax_enable_x64", False)

from qugen.main.generator.base_model_handler import BaseModelHandler
from qugen.main.generator.quantum_circuits.continuous_circuits import get_qnode
from qugen.main.data.helper import kl_divergence_from_data
from qugen.main.data.helper import CustomDataset
from qugen.main.discriminator.discriminator_for_continuous_qgan import Discriminator
from qugen.main.data.data_handler import PITNormalizer, MinMaxNormalizer


class ContinuousQGANModelHandler(BaseModelHandler):
    """
    Parameters:

    """

    def __init__(self):
        """Initialize the parameters specific to this model handler by assigning defaults to all attributes which should immediately be available across all methods."""
        super().__init__()
        self.n_qubits = None
        self.circuit_depth = None
        self.weights = None
        self.generator_weights = None
        self.discriminator_weights = None
        self.performed_trainings = 0
        self.circuit = None
        self.generator = None
        self.num_params = None
        self.slower_progress_update = None
        self.normalizer = None

    def build(
        self,
        model_name: str,
        data_set: str,
        n_qubits: int = 2,
        circuit_depth: int = 1,
        random_seed: int = 42,
        transformation: str = "pit",
        save_artifacts=True,
        slower_progress_update=False,
    ) -> BaseModelHandler:
        """Build the continuous qgan model.
        This defines the architecture of the model, including the circuit ansatz, data transformation and whether the artifacts are saved.

        Args:
            model_name (int): The name which will be used to save the data to disk.
            data_set: The name of the data set which gets as part of the model name
            n_qubits (int, optional): Number of qubits. Defaults to 2.
            circuit_depth (int, optional): Number of repetitions of qml.StronglyEntanglingLayers. Defaults to 1.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
            transformation (str, optional): Type of normalization, either "minmax" or "pit". Defaults to "pit".
            save_artifacts (bool, optional): Whether to save the artifacts to disk. Defaults to True.
            slower_progress_update (bool, optional): Controls how often the progress bar is updated. If set to True, update every 10 seconds at most, otherwise use tqdm defaults. Defaults to False.

        Returns:
            BaseModelHandler: Return the built model handler. It is not strictly necessary to overwrite the existing variable with this
            since all changes are made in place.
        """
        self.slower_progress_update = slower_progress_update
        self.save_artifacts = save_artifacts
        self.random_key = jax.random.PRNGKey(random_seed)
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.transformation = transformation
        time_str = str(time.time()).encode("utf-8")
        uniq = hashlib.md5(time_str).hexdigest()[:4]

        self.data_set = data_set

        circuit_type = "continuous"
        self.model_name = f"{model_name}_{self.data_set}_{self.transformation}_qgan_{uniq}"
        print(f"{self.model_name=}")
        self.path_to_models = "experiments/" + self.model_name

        self.metadata = dict(
            {
                "model_name": self.model_name,
                "data_set": self.data_set,
                "n_qubits": self.n_qubits,
                "circuit_type": circuit_type,
                "circuit_depth": self.circuit_depth,
                "transformation": self.transformation,
                "discriminator": "digital",
                "training_data": {},
            }
        )
        if save_artifacts:
            os.makedirs(self.path_to_models)
            with open(self.path_to_models + "/" + "meta.json", "w") as fp:
                json.dump(self.metadata, fp)
        if self.transformation == "minmax":
            self.normalizer = MinMaxNormalizer()
        elif self.transformation == "pit":
            self.normalizer = PITNormalizer()
        else:
            raise ValueError("Transformation value must be either 'minmax' or 'pit'")
        if self.generator is None:
            self.generator, self.num_params = get_qnode(self.circuit_depth, self.n_qubits)
        return self

    def save(self, file_path: Path, overwrite: bool = True) -> BaseModelHandler:
        """Save the generator and discriminator weights to disk.

        Args:
            file_path (Path): The paths where the pickled tuple of generator and discriminator weights will be placed.
            overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to True.

        Returns:
            BaseModelHandler: The model, unchanged.
        """
        if overwrite or not os.path.exists(file_path):
            with open(file_path, "wb") as file:
                pickle.dump((self.generator_weights, self.discriminator_weights), file)
        return self

    def reload(self, model_name: str, epoch: int) -> BaseModelHandler:
        """Reload the model from the artifacts including the parameters for the generator and the discriminator,
        the metadata and the data transformation file (reverse lookup table or original min and max of the training data).

        Args:
            model_name (str): The name of the model to reload.
            epoch (int): The epoch to reload.

        Returns:
            BaseModelHandler: The reloaded model, but changes have been made in place as well.
        """
        weights_file = "experiments/" + model_name + "/" + "parameters_training_iteration={0}.pickle".format(str(epoch))
        meta_file = "experiments/"+ model_name + "/" +  "meta.json"
        reverse_file = "experiments/" + model_name + "/" + 'reverse_lookup.npy'

        with open(weights_file, "rb") as file:
            self.generator_weights, self.discriminator_weights = pickle.load(file)
        with open(meta_file, "r") as file:
            self.metadata = json.load(file)

        self.reverse_lookup = jnp.load(reverse_file)
        self.n_qubits = self.metadata["n_qubits"]
        self.transformation = self.metadata["transformation"]
        self.circuit_depth = self.metadata["circuit_depth"]
        self.performed_trainings = len(self.metadata["training_data"])
        self.random_key = jax.random.PRNGKey(2)
        self.path_to_models =  "experiments/" + self.metadata["model_name"]

        if self.normalizer is None:
            if self.transformation == "minmax":
                self.normalizer = MinMaxNormalizer()
            elif self.transformation == "pit":
                self.normalizer = PITNormalizer()
            else:
                raise ValueError("Transformation value must be either 'minmax' or 'pit'")
        self.normalizer.reverse_lookup = self.reverse_lookup
        if self.generator is None:
            self.generator, self.num_params = get_qnode(self.circuit_depth, self.n_qubits)
        return self

    def train(
        self,
        train_dataset_original_space: np.array,
        n_epochs: int,
        initial_learning_rate_generator: float,
        initial_learning_rate_discriminator: float,
        batch_size=None,
    ) -> BaseModelHandler:
        """Train the continuous QGAN.

        Args:
            train_dataset_original_space (np.array): The training data in the original space.
            n_epochs (int): Technically, we are not passing the number of passes through the training data, but the number of iterations of the training loop.
            initial_learning_rate_generator (float, optional): Learning rate for the quantum generator.
            initial_learning_rate_discriminator (float, optional): Learning rate for the classical discriminator.
            batch_size (int, optional): Batch size. Defaults to None, and the whole training data is used in each iteration.
            
        Raises:
            ValueError: Raises ValueError if the training dataset has dimension (number of columns) not equal to 2 or 3.

        Returns:
            BaseModelHandler: The trained model.
        """
        self.batch_size = len(train_dataset_original_space) if batch_size is None else batch_size
        
        if self.performed_trainings == 0:
            self.previous_trained_epochs = 0 
        else:
            self.previous_trained_epochs = sum([self.metadata["training_data"][str(i)]["n_epochs"] for i in range(self.performed_trainings)])

        training_data = {}
        training_data["n_epochs"] = n_epochs
        training_data["batch_size"] = self.batch_size
        training_data[
            "initial_learning_rate_generator"
        ] = initial_learning_rate_generator
        training_data[
            "initial_learning_rate_discriminator"
        ] = initial_learning_rate_discriminator
        self.metadata["training_data"][str(self.performed_trainings)] = training_data
        self.performed_trainings += 1

        train_dataset = self.normalizer.fit_transform(train_dataset_original_space)
        train_dataset = (train_dataset - 0.5) * 2
        self.reverse_lookup = self.normalizer.reverse_lookup

        if self.save_artifacts:
            with open(self.path_to_models + "/" + "meta.json", "w+") as file:
                json.dump(self.metadata, file)

            jnp.save(self.path_to_models + "/" + "reverse_lookup.npy", self.reverse_lookup)

        X_train = CustomDataset(train_dataset.astype("float32"))
        D = Discriminator()
        D.apply = jax.jit(D.apply)
        epsilon = 1e-10
        v_qnode = jax.vmap(self.generator, in_axes=(0, None))

        def cost_fn_discriminator(z, X, generator_weights, discriminator_weights):
            G_sample = v_qnode(z, generator_weights)
            D_fake = D.apply(discriminator_weights, G_sample)
            D_real = D.apply(discriminator_weights, X)
            loss_1 = -jnp.mean(jnp.log(D_real + epsilon))
            loss_2 = -jnp.mean(jnp.log(1.0 - D_fake + epsilon))
            D_loss = loss_1 + loss_2
            return D_loss

        def cost_fn_generator(z, generator_weights, discriminator_weights):
            G_sample = v_qnode(z, generator_weights)
            D_fake = D.apply(discriminator_weights, G_sample)
            G_loss = -jnp.mean(jnp.log(D_fake + epsilon))  # Vanilla GAN
            return G_loss

        kl_list_transformed_space = []
        it_list = []
        progress = tqdm(
            range(n_epochs), mininterval=10 if self.slower_progress_update else None
        )
        synthetic_transformed_space = 0  # init so its a global var

        self.random_key, subkey1, subkey2, subkey3 = jax.random.split(
            self.random_key, num=4
        )
        w_shape = (self.circuit_depth, 1, self.n_qubits, 3)

        # Currently it is not possible that one type of weight is None while the other is not, but keep the
        # if-statements separate for now anyways in case we want to do that in the future.
        if self.generator_weights is None:
            self.generator_weights = (
                jax.random.normal(subkey1, shape=w_shape) * 2 * np.pi - np.pi
            )
        if self.discriminator_weights is None:
            x = jax.random.uniform(subkey2, (self.n_qubits,))  # Dummy input
            self.discriminator_weights = D.init(subkey3, x)

        optimizer_generator = optax.adam(initial_learning_rate_generator)
        optimizer_state_g = optimizer_generator.init(self.generator_weights)

        optimizer_discriminator = optax.adam(initial_learning_rate_discriminator)
        optimizer_state_d = optimizer_discriminator.init(self.discriminator_weights)

        discriminator_training_steps = 1

        self.random_key, subkey = jax.random.split(self.random_key)
        z = jax.random.normal(subkey, (self.batch_size, self.n_qubits))
        if self.batch_size != len(train_dataset):
            self.random_key, subkey = jax.random.split(self.random_key)

        for it in progress:
            if self.save_artifacts:
                self.save(
                    f"{self.path_to_models}/parameters_training_iteration={it + self.previous_trained_epochs }.pickle",
                    overwrite=False,
                 )
            X = X_train.next_batch(self.batch_size)

            for _ in range(discriminator_training_steps):
                cost_discriminator, grad = jax.value_and_grad(
                    lambda w: cost_fn_discriminator(z, X, self.generator_weights, w)
                )(self.discriminator_weights)

                updates, optimizer_state_d = optimizer_discriminator.update(
                    grad, optimizer_state_d
                )
                self.discriminator_weights = optax.apply_updates(
                    self.discriminator_weights, updates
                )

            cost_generator, grad = jax.value_and_grad(
                lambda w: cost_fn_generator(z, w, self.discriminator_weights)
            )(self.generator_weights)

            updates, optimizer_state_g = optimizer_generator.update(
                grad, optimizer_state_g
            )
            self.generator_weights = optax.apply_updates(
                self.generator_weights, updates
            )

            # Update progress bar postfix and calculate KL-divergence in transformed and original space
            if it % 10 == 0:
                synthetic_transformed_space = self.predict_transform(
                    n_samples=len(train_dataset)
                )
                kl_transformed_space = kl_divergence_from_data(
                    (train_dataset + 1)/ 2,
                    synthetic_transformed_space,
                    number_bins=16,
                    bin_range=[[0, 1] for _ in range(self.n_qubits)],
                    dimension=self.n_qubits,
                )

                progress.set_postfix(
                    loss_generator=cost_generator,
                    loss_discriminator=cost_discriminator,
                    kl_transformed_space=kl_transformed_space,
                    major_layer=self.circuit_depth,
                    refresh=False if self.slower_progress_update else None,
                )

                kl_list_transformed_space.append(kl_transformed_space)
                it_list.append(it)
        if self.save_artifacts:
            self.save(
                f"{self.path_to_models}/parameters_training_iteration={it + self.previous_trained_epochs}.pickle",
                overwrite=False,
            )
        return self

    def predict(
        self,
        n_samples: int = 32,
    ) -> np.array:
        """Generate samples from the trained model and perform the inverse of the data transformation
        which was used to transform the training data to be able to compute the KL-divergence in the original space.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 32.

        Returns:
            np.array: Array of samples of shape (n_samples, sample_dimension).
        """
        samples_transformed = self.predict_transform(n_samples)

        if self.transformation == "pit":
            self.transformer = PITNormalizer()
        elif self.transformation == "minmax":
            self.transformer = MinMaxNormalizer()
        self.transformer.reverse_lookup = self.reverse_lookup
        samples = self.transformer.inverse_transform(samples_transformed)
        return samples

    def predict_transform(
        self,
        n_samples: int = 32,
    ) -> np.array:
        """Generate samples from the trained model in the transformed space (the n-dimensional unit cube).

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
        noise = jax.random.normal(subkey, (n_samples, self.n_qubits))
        v_qnode = jax.vmap(lambda inpt: self.generator(inpt, self.generator_weights))
        samples_transformed = (v_qnode(noise) + 1) / 2
        samples_transformed = np.asarray(samples_transformed)

        return samples_transformed

    def sample(self, n_samples: int = 32):
        """Generate samples from the trained model.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 32.

        Returns:
            np.array: Array of samples of shape (n_samples, sample_dimension).
        """
        return self.predict(n_samples)
