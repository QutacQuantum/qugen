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

import glob
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from qugen.main.data.helper import kl_divergence_from_data
import matplotlib.pyplot as plt


class BaseModelHandler(ABC):
    """
    It implements the interface for each of the models handlers (continuous QGAN/QCBM and discrete QGAN/QCBM),
    which includes building the models, training them, saving and reloading them, and generating samples from them.
    """

    def __init__(self):
        """"""
        self.device_configuration = None

    @abstractmethod
    def build(self, *args, **kwargs) -> "BaseModelHandler":
        """
        Define the architecture of the model. Weights initialization is also typically performed here.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, file_path: Path, overwrite: bool = True) -> "BaseModelHandler":
        """
        Saves the model weights to a file.

        Parameters:
            file_path (pathlib.Path): destination file for model weights
            overwrite (bool): Flag indicating if any existing file at the target location should be overwritten
        """
        raise NotImplementedError

    @abstractmethod
    def reload(self, file_path: Path) -> "BaseModelHandler":
        """
        Loads the model from a set of weights.

        Parameters:
            file_path (pathlib.Path): source file for the model weights
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, *args) -> "BaseModelHandler":
        """
        Perform training of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args) -> np.array:
        """
        Draw samples from the model.
        """
        raise NotImplementedError

    def evaluate(
        self, train_dataset_original_space: np.ndarray, number_bins=16
    ) -> pd.DataFrame:
        parameters_all_training_iterations = glob.glob(
            f"{self.path_to_models}/parameters_training_iteration=*"
        )
        it_list = []
        kl_list_transformed_space = []
        kl_list_original_space = []
        train_dataset_transformed_space = self.normalizer.transform(
            train_dataset_original_space
        )
        dimension = train_dataset_original_space.shape[1]
        progress = tqdm(range(len(parameters_all_training_iterations)))
        progress.set_description("Evaluating")
        best_kl_original_space = np.inf
        best_samples_original_space = None
        for it in progress:
            parameters_path = parameters_all_training_iterations[it]
            iteration = re.search(
                "parameters_training_iteration=(.*).(pickle|npy)",
                os.path.basename(parameters_path),
            ).group(1)
            it_list.append(iteration)
            self.reload(self.model_name, int(iteration))
            synthetic_transformed_space = self.predict_transform(
                n_samples=100000 #len(train_dataset_original_space)
            )
            synthetic_original_space = self.normalizer.inverse_transform(
                synthetic_transformed_space
            )
            kl_transformed_space = kl_divergence_from_data(
                train_dataset_transformed_space,
                synthetic_transformed_space,
                number_bins=number_bins,
                bin_range=[[0, 1] for _ in range(dimension)],
                dimension=dimension,
            )
            kl_list_transformed_space.append(kl_transformed_space)

            kl_original_space = kl_divergence_from_data(
                train_dataset_original_space,
                synthetic_original_space,
                number_bins=number_bins,
                bin_range=list(
                    zip(
                        train_dataset_original_space.min(axis=0),
                        train_dataset_original_space.max(axis=0),
                    )
                ),
                dimension=dimension,
            )
            kl_list_original_space.append(kl_original_space)

            if kl_original_space < best_kl_original_space:
                best_kl_original_space = kl_original_space
                best_samples_original_space = synthetic_original_space

            progress.set_postfix(
                kl_original_space=kl_original_space,
                kl_transformed_space=kl_transformed_space,
                refresh=False,
            )

        if dimension in [2,3]:
            fig = plt.figure()
            fig.suptitle(f"{self.path_to_models}, KL={best_kl_original_space}")
            samples_idx = np.random.choice(len(best_samples_original_space), 1000)
            samples = best_samples_original_space[samples_idx]
            if dimension == 2:
                ax = fig.add_subplot()
                ax.scatter(samples[:, 0], samples[:, 1])
            elif dimension == 3:
                ax = fig.add_subplot(projection='3d')
                ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])
                ax.view_init(elev=10., azim=20)
            plt.savefig(f"{self.path_to_models}/scatterplot_best_samples_original_space.png")
        elif dimension > 3:
            print("Not Creating a Scatter plot as the dimension is greater than 3.")
        else:
            raise ValueError
        
        it_list = list(map(lambda x: int(x), it_list))
        kl_results = pd.DataFrame(
            {
                "iteration": it_list,
                "kl_transformed_space": np.array(kl_list_transformed_space).astype(float),
                "kl_original_space": np.array(kl_list_original_space).astype(float),
            }
        )
        kl_results = kl_results.sort_values(by=["iteration"])
        kl_results.to_csv(f"{self.path_to_models}/kl_results.csv", index=False)
        return kl_results
