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
import warnings
from itertools import chain
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
from tqdm import tqdm

from qugen.main.generator.quantum_circuits.discrete_generator_pennylane import compute_gradient_JAX
from qugen.main.generator.base_model_handler import BaseModelHandler
from qugen.main.data.helper import CustomDataset
from qugen.main.data.data_handler import PITNormalizer, MinMaxNormalizer
from qugen.main.data.helper import kl_divergence
from qugen.main.discriminator.discriminator import Discriminator_JAX
from qugen.main.data.discretization import compute_discretization

import matplotlib.pyplot as plt
import matplotlib as mpl


jax.config.update("jax_enable_x64", True)
mpl.use("Agg")


class DiscreteQGANModelHandler(BaseModelHandler):

    def __init__(self):
        """Initialize the parameters specific to this model handler by assigning defaults to all attributes which should immediately be available across all methods."""
        super().__init__()
        self.n_qubits = None
        self.n_registers = None
        self.circuit_depth = None
        self.weights = None
        self.generator = None
        self.num_generator_params = None
        self.circuit = None
        self.n_epochs = None
        self.generator_weights = None
        self.discriminator_weights = None
        self.random_key = None
        self.reverse_lookup = None
        self.save_artifacts = None
        self.slower_progress_update = None
        self.normalizer = None

    def build(
        self,
        model_name: str,
        data_set_name: str,
        n_qubits=8,
        n_registers=2,
        circuit_depth=1,
        random_seed=42,
        transformation="pit",
        circuit_type="copula",
        save_artifacts=True,
        slower_progress_update=False,
    ) -> BaseModelHandler:
        """Build the discrete QGAN model.
        This defines the architecture of the model, including the circuit ansatz, data transformation and whether the artifacts are saved.

        Args:
            model_name (str): The name which will be used to save the data to disk.
            data_set_name (str): The name of the data set which is set as part of the model name
            n_qubits (int, optional): Number of qubits. Defaults to 8.
            n_registers (int): Number of dimensions of the data.
            circuit_depth (int, optional): Number of repetitions of qml.StronglyEntanglingLayers. Defaults to 1.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
            transformation (str, optional): Type of normalization, either "minmax" or "pit". Defaults to "pit".
            circuit_type (string, optional): name of the circuit anstaz to be used for the QGAN, either "copula" or "standard". Defaults to "copula"
            save_artifacts (bool, optional): Whether to save the artifacts to disk. Defaults to True.
            slower_progress_update (bool, optional): Controls how often the progress bar is updated. If set to True, update every 10 seconds at most, otherwise use tqdm defaults. Defaults to False.

        Returns:
            BaseModelHandler: Return the built model handler. It is not strictly necessary to overwrite the existing variable with this
            since all changes are made in place.
        """
        self.slower_progress_update = slower_progress_update
        self.n_qubits = n_qubits
        self.n_registers = n_registers
        self.circuit_depth = circuit_depth
        self.data_set_name = data_set_name
        self.transformation = transformation
        self.circuit_type = circuit_type
        self.performed_trainings = 0
        self.save_artifacts = save_artifacts

        time_str = str(time.time()).encode('utf-8')
        uniq = hashlib.md5(time_str).hexdigest()[:4]

        self.model_name = model_name + '_' + self.data_set_name + '_' + self.circuit_type + '_' + self.transformation+ '_' + 'qgan_' + uniq

        self.device = 'cpu'
        self.beta_1 = 0.5
        self.real_label = 1.
        self.fake_label = 0.
        self.n_samples = 10000

        self.path_to_models = "experiments/" + self.model_name

        self.metadata = dict({
            'model_name': self.model_name,
            'n_qubits': self.n_qubits,
            'n_registers': self.n_registers,
            'circuit_type': self.circuit_type,
            'circuit_depth': self.circuit_depth,
            'transformation': self.transformation,
            'data_set ': self.data_set_name,
            'n_epochs': self.n_epochs,
            'discriminator': 'digital',
            "training_data": {},
        })

        # save artifacts only when save_artifacts flag is true, used for testing
        if save_artifacts:
            # create experiments folder
            os.makedirs('experiments/' + self.model_name)
            print('model_name', self.model_name)
            with open(
                    self.path_to_models + "/" + "meta.json", "w"
            ) as fp:
                json.dump(self.metadata, fp)

        # jax specific
        self.random_key = jax.random.PRNGKey(random_seed)

        self.D = Discriminator_JAX()
        self.D.apply = jax.jit(self.D.apply)
        self.random_key, subkey1, subkey2 = jax.random.split(self.random_key, num=3)
        self.discriminator_weights = self.D.init(
            subkey2,
            jax.random.uniform(
                subkey1,
                (
                    1,
                    self.n_qubits,
                ),
            ),
        )  # Use dummy input for init

        if self.transformation == 'minmax':
            self.normalizer = MinMaxNormalizer(epsilon=1e-6)
        elif self.transformation == 'pit':
            self.normalizer = PITNormalizer(epsilon=1e-6)
        else:
            raise ValueError("Transformation value must be either 'minmax' or 'pit'")

        if self.circuit_type == 'copula':
            from qugen.main.generator.quantum_circuits.discrete_generator_pennylane \
                import discrete_copula_circuit_JAX as get_generator
        elif self.circuit_type == 'standard':
            from qugen.main.generator.quantum_circuits.discrete_generator_pennylane \
                import discrete_standard_circuit_JAX as get_generator
        else:
            raise ValueError("Circuit value must be either 'standard' or 'copula'")
        self.generator, self.num_generator_params = get_generator(self.n_qubits, self.n_registers, self.circuit_depth)
        self.random_key, subkey = jax.random.split(self.random_key)
        
        # Draw from interval [0, pi) because that is how it was before
        self.generator_weights = jax.random.uniform(subkey, shape=(self.num_generator_params,)) * jnp.pi
        print(f"{self.num_generator_params=}")


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


    def reload(
        self, model_name: str, epoch: int, random_seed: Optional[int] = None
    ) -> BaseModelHandler:
        """Reload the model from the artifacts including the parameters for the generator and the discriminator,
        the metadata and the data transformation file (reverse lookup table or original min and max of the training data).

        Args:
            model_name (str): The name of the model to reload.
            epoch (int): The epoch to reload.
            random_seed (int, Optional): Specify a random seed for reproducibility.

        Returns:
            BaseModelHandler: The reloaded model, but changes have been made in place as well.
        """        
        self.model_name = model_name
        self.path_to_models = "experiments/" + self.model_name
        weights_file = "experiments/" + model_name + "/" + "parameters_training_iteration={0}.pickle".format(str(epoch))
        meta_file = "experiments/"+ model_name + "/" + "meta.json"
        reverse_file = "experiments/" + model_name + "/" + 'reverse_lookup.npy'

        with open(weights_file, "rb") as file:
            self.generator_weights, self.discriminator_weights = pickle.load(file)
        with open(meta_file, 'r') as f:
            self.metadata = json.load(f)
        self.reverse_lookup = jnp.load(reverse_file)

        self.n_qubits = self.metadata["n_qubits"]
        self.transformation = self.metadata["transformation"]
        self.circuit_depth = self.metadata["circuit_depth"]
        self.performed_trainings = len(self.metadata["training_data"])
        self.n_registers = self.metadata['n_registers']
        self.circuit_type = self.metadata['circuit_type']

        if random_seed is None:
            if self.random_key is None:
                self.random_key = jax.random.PRNGKey(2)
        else:
            if self.random_key is not None:
                warnings.warn(
                    "Random state already initialized in the model handler, but a random_seed was specified when reloading. "
                    "Re-initializing with the random_seed."
                )
            self.random_key = jax.random.PRNGKey(random_seed)


        if self.normalizer is None:
            if self.transformation == 'minmax':
                self.normalizer = MinMaxNormalizer(epsilon=1e-6)
            elif self.transformation == 'pit':
                self.normalizer = PITNormalizer(epsilon=1e-6)
            else:
                raise ValueError("Transformation value must be either 'minmax' or 'pit'")
        self.normalizer.reverse_lookup = self.reverse_lookup

        if self.generator is None:
            if self.circuit_type == 'copula':
                from qugen.main.generator.quantum_circuits.discrete_generator_pennylane import \
                    discrete_copula_circuit_JAX as get_generator
            elif self.circuit_type == 'standard':
                from qugen.main.generator.quantum_circuits.discrete_generator_pennylane import \
                    discrete_standard_circuit_JAX as get_generator
            else:
                raise ValueError("Circuit value must be either 'standard' or 'copula'")
            self.generator, self.num_generator_params = get_generator(self.n_qubits, self.n_registers, self.circuit_depth)
        return self

    def train(
        self,
        train_dataset: np.array,
        n_epochs: int,
        initial_learning_rate_generator: float,
        initial_learning_rate_discriminator: float,
        batch_size = 1000,
    ) -> BaseModelHandler:
        """Train the discrete QGAN.

        Args:
            train_dataset (np.array): The training data in the original space.
            n_epochs (int): Technically, we are not passing the number of passes through the training data, but the number of iterations of the training loop.
            initial_learning_rate_generator (float, optional): Learning rate for the quantum generator.
            initial_learning_rate_discriminator (float, optional): Learning rate for the classical discriminator.
            batch_size (int, optional): Batch size. Defaults to None, and the whole training data is used in each iteration.
            
        Raises:
            ValueError: Raises ValueError if the training dataset has dimension (number of columns) not equal to 2 or 3.

        Returns:
            BaseModelHandler: The trained model.
        """

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        if self.performed_trainings == 0:
            self.previous_trained_epochs = 0
        else:
            self.previous_trained_epochs = sum([self.metadata["training_data"][str(i)]["n_epochs"] for i in range(self.performed_trainings)])
        training_data = {}
        training_data["n_epochs"] = self.n_epochs
        training_data["batch_size"] = self.batch_size
        training_data["learning_rate_generator"] = initial_learning_rate_generator
        training_data["learning_rate_discriminator"] = initial_learning_rate_discriminator
        self.metadata["training_data"][str(self.performed_trainings)] = training_data
        self.performed_trainings += 1

        train_dataset = self.normalizer.fit_transform(train_dataset)
        self.reverse_lookup = self.normalizer.reverse_lookup
        if self.save_artifacts:
            with open(self.path_to_models + "/" + "meta.json", "w+") as file:
                json.dump(self.metadata, file)

            jnp.save(self.path_to_models + "/" + 'reverse_lookup.npy', self.reverse_lookup)


        self.dict_bins = compute_discretization(self.n_qubits, self.n_registers)

        n = 2 ** (self.n_qubits // self.n_registers)

        nns = tuple(n for _ in range(self.n_registers))
        nns_nq = nns + tuple((self.n_qubits,))

        inverse_bins = np.zeros(nns_nq)
        for key, value in self.dict_bins.items():
            id_n = value[0]
            inverse_bins[id_n] = jnp.array([int(bit) for bit in key])

        coordinates = np.floor(train_dataset * n).astype(int)
        train_dataset = [
            inverse_bins[tuple([xy[ii] for ii in range(self.n_registers)])]
            for xy in coordinates
        ]
        train_dataset = jnp.array(train_dataset).astype(jnp.float32)
        distribution_pit = np.zeros(nns)
        for xy in coordinates:
            indices = tuple(xy[ii] for ii in range(self.n_registers))
            distribution_pit[indices] += 1
        distribution_pit /= np.sum(distribution_pit)
        distribution_pit = jnp.array(distribution_pit)

        optimizer_discriminator = optax.adam(
            learning_rate=initial_learning_rate_discriminator,
            b1=self.beta_1,
            b2=0.999,
        )
        optimizer_state_d = optimizer_discriminator.init(self.discriminator_weights)
        optimizer_generator = optax.sgd(learning_rate=initial_learning_rate_generator)
        self.random_key, subkey = jax.random.split(self.random_key)
        optimizer_state_g = optimizer_generator.init(self.generator_weights)
        kl_list_transformed_space = []
        it_list = []

        # create shifts in advance, leads to less code at application
        elementary_shift = 1
        shifts = [
            [elementary_shift * e_i, -elementary_shift * e_i]
            for e_i in jnp.eye(self.generator_weights.size)
        ]
        shifts = list(chain(*shifts))
        shifts = [shift.reshape(self.generator_weights.shape) for shift in shifts]
        parameters = []

        epsilon = 1e-10

        X_train = CustomDataset(train_dataset.astype("float32"))

        def cost_fn_discriminator(X, generator_weights, discriminator_weights):
            self.random_key, subkey = jax.random.split(self.random_key)
            G_samples = self.generator(
                subkey,
                generator_weights,
                n_shots=len(X),
            )
            D_fake = self.D.apply(discriminator_weights, G_samples)
            D_real = self.D.apply(discriminator_weights, X)
            loss_1 = -jnp.mean(jnp.log(D_real + epsilon))
            loss_2 = -jnp.mean(jnp.log(1.0 - D_fake + epsilon))
            D_loss = loss_1 + loss_2
            return D_loss

        def cost_fn_generator(X, generator_weights, discriminator_weights):
            self.random_key, subkey = jax.random.split(self.random_key)
            G_samples = self.generator(
                subkey,
                weights=generator_weights,
                n_shots=len(X),
            )
            D_fake = self.D.apply(discriminator_weights, G_samples)
            G_loss = -jnp.mean(jnp.log(D_fake + epsilon))  # Vanilla GAN
            return G_loss

        progress = tqdm(range(n_epochs), mininterval=10 if self.slower_progress_update else None)

        for it in progress:
            if self.save_artifacts:
                self.save(
                    f"{self.path_to_models}/parameters_training_iteration={it + self.previous_trained_epochs }.pickle",
                    overwrite=False,
                )
            data = X_train.next_batch(self.batch_size)

            discriminator_training_steps = 1  # How many times is the discriminator updates per generator update
            for _ in range(discriminator_training_steps):
                cost_discriminator, grad_d = jax.value_and_grad(
                    lambda w: cost_fn_discriminator(data, self.generator_weights, w)
                )(self.discriminator_weights)
                updates, optimizer_state_d = optimizer_discriminator.update(
                    grad_d, optimizer_state_d
                )
                self.discriminator_weights = optax.apply_updates(
                    self.discriminator_weights, updates
                )
            # This is the method using the old manual gradient
            cost_generator = cost_fn_generator(
                data, self.generator_weights, self.discriminator_weights
            )
            self.random_key, *subkeys = jax.random.split(self.random_key, num=len(shifts) + 1)
            G_samples = [
                self.generator(
                    subkey,
                    self.generator_weights + parameter_shift,
                    n_shots=self.batch_size,
                )
                for subkey, parameter_shift in zip(subkeys, shifts)
            ]

            grad_g = compute_gradient_JAX(
                G_samples, self.D, self.discriminator_weights
            )
            grad_g = grad_g.reshape(self.generator_weights.shape)


            updates, optimizer_state_g = optimizer_generator.update(
                grad_g, optimizer_state_g
            )
            self.generator_weights = optax.apply_updates(
                self.generator_weights, updates
            )


            parameters.append(self.generator_weights.copy())

            # Update progress bar postfix and calculate KL-divergence in transformed and original space
            if it % 100 == 0:
                self.random_key, subkey = jax.random.split(self.random_key)
                samples = self.generator(
                subkey,
                self.generator_weights,
                n_shots=self.n_samples,
            )
                # Split the binary strings for each dimension into separate arrays
                samples_dims = []
                for dim in range(self.n_registers):
                    samples_dims.append(
                        samples[
                            :,
                            dim
                            * self.n_qubits
                            // self.n_registers : (dim + 1)
                            * self.n_qubits
                            // self.n_registers,
                        ]
                    )
                # Calculate the decimal representation of the binary strings
                indices_fn = jax.vmap(
                    lambda sample: jnp.dot(2 ** jnp.arange(0, sample.size), sample[::-1])
                )
                indices = jnp.vstack([indices_fn(dim) for dim in samples_dims]).T
                # compute accuracy
                n = 2 ** (self.n_qubits // self.n_registers)

                distribution_generator = jnp.histogramdd(
                    indices, bins=[jnp.arange(0, n+1) for _ in range(self.n_registers)],
                )[0]
                distribution_generator = distribution_generator / jnp.sum(
                    distribution_generator
                )

                kl_transformed_space = kl_divergence(distribution_pit, distribution_generator)
                kl_list_transformed_space.append(kl_transformed_space)

                progress.set_postfix(
                    loss_generator=cost_generator,
                    loss_discriminator=cost_discriminator,
                    kl_transformed_space=kl_transformed_space,
                    major_layer=self.circuit_depth,
                    refresh=False if self.slower_progress_update else None,
                )

                it_list.append(it)
        if self.save_artifacts:
            self.save(
                f"{self.path_to_models}/parameters_training_iteration={it + 1 + self.previous_trained_epochs}.pickle",
                overwrite=True,
            )
            if not self.slower_progress_update and self.n_registers == 2:
                fig, ax = plt.subplots(1, 2)

                ax[0].imshow(distribution_pit, origin="lower", interpolation=None)

                ax[1].imshow(
                    np.array(distribution_generator), origin="lower", interpolation=None
                )

                plt.savefig(
                    "experiments/"
                    + self.model_name
                    + "/"
                )
                plt.close(fig)

        parameters = []

        return self

    def predict(
        self,
        n_samples: int = 32
    ) -> np.array:
        """Generate samples from the trained model and perform the inverse of the data transformation
        which was used to transform the training data to be able to compute the KL-divergence in the original space.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 32.

        Returns:
            np.array: Array of samples of shape (n_samples, sample_dimension).
        """
        samples_transformed = self.predict_transform(n_samples)

        if self.transformation == 'pit':
            self.normalizer = PITNormalizer(epsilon=1e-6)
        elif self.transformation == 'minmax':
            self.normalizer = MinMaxNormalizer(epsilon=1e-6)

        self.normalizer.reverse_lookup = self.reverse_lookup
        samples = self.normalizer.inverse_transform(samples_transformed)
        return samples

    def predict_transform(
        self,
        n_samples: int = 32,
        noisy=True,
    ) -> np.array:
        """Generate samples from the trained model in the transformed space (the n-dimensional unit cube).

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 32.
            noisy (bool, optional): Controls the generation of uniformly distributed noise around each generated sample to help with discretization limitations.

        Returns:
            np.array: Array of samples of shape (n_samples, sample_dimension).
        """        

        self.random_key, subkey = jax.random.split(self.random_key)
        samples = self.generator(
            subkey,
            self.generator_weights,
            n_shots=n_samples,
        )
        # Split the binary strings for each dimension into separate arrays
        samples_dims = []
        for dim in range(self.n_registers):
            samples_dims.append(
                samples[
                    :,
                    dim
                    * self.n_qubits
                    // self.n_registers : (dim + 1)
                    * self.n_qubits
                    // self.n_registers,
                ]
            )
        # Calculate the decimal representation of the binary strings
        indices_fn = jax.vmap(
            lambda sample: jnp.dot(2 ** jnp.arange(0, sample.size), sample[::-1])
        )

        n = 2 ** (self.n_qubits // self.n_registers)
        indices = jnp.vstack([indices_fn(dim) / n for dim in samples_dims]).T
        if noisy:
            # Random seed is explicit in JAX
            self.random_key, subkey = jax.random.split(self.random_key)
            noises = jax.random.uniform(subkey, shape=(n_samples, self.n_registers))
            self.dict_bins = compute_discretization(self.n_qubits, self.n_registers)
            width = 1 / (np.power(len(self.dict_bins), (1 / self.n_registers)))
            samples = indices + 1.0/(2*n) + (noises - 0.5) * width
        else:
            samples = indices

        return samples
