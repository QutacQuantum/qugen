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

import copy
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from jax import config

# config.update("jax_debug_nans", True)

from qugen.main.generator.discrete_qcbm_model_handler import (
    DiscreteQCBMModelHandler,
)
from qugen.main.generator.discrete_qgan_model_handler import (
    DiscreteQGANModelHandler,
)
from qugen.main.generator.continuous_qcbm_model_handler import (
    ContinuousQCBMModelHandler,
)
from qugen.main.generator.continuous_qgan_model_handler import (
    ContinuousQGANModelHandler,
)
from qugen.main.data.data_handler import load_data

minimum_kl_data_path = f"data"


def helper_test_configuration(
    model, configuration, reference_data, fixed_hyperparameters
):
    name, data_set_name, dimension, circuit_depth, normalization = configuration

    data_set_path = f"../../apps/logistics/training_data/{data_set_name}_{dimension}D"
    data = load_data(data_set_path)[0]
    print(f"{len(data)=}")

    if "qcbm" in name:
        model.train(
            data,
            n_epochs=fixed_hyperparameters["n_epochs"],
            batch_size=fixed_hyperparameters["batch_size"],
            hist_samples=fixed_hyperparameters["hist_samples"],
        )
    elif "qgan" in name:
        model.train(
            data,
            n_epochs=fixed_hyperparameters["n_epochs"],
            batch_size=fixed_hyperparameters["batch_size"],
            initial_learning_rate_generator=fixed_hyperparameters[
                "learning_rate_generator"
            ],
            initial_learning_rate_discriminator=fixed_hyperparameters[
                "learning_rate_discriminator"
            ],
        )

    try:
        kl_original_reference = reference_data["kl_original"].values
    except KeyError:
        kl_original_reference = reference_data["kl"].values

    evaluation_df = model.evaluate(data)

    # get the row with the minimum kl
    minimum_kl_data = evaluation_df.loc[evaluation_df["kl_original_space"].idxmin()]
    minimum_kl_calculated = minimum_kl_data["kl_original_space"]
    minimum_kl_transformed = evaluation_df.loc[evaluation_df["kl_transformed_space"].idxmin()]
    minimum_kl_transformed_iteration = minimum_kl_transformed["iteration"]

    # Check if minimum_kl_calculated if within one standard deviation of the minimum_kl from the reference data
    # assert (
    # #     # np.abs(minimum_kl_calculated - kl_original_reference.mean())
    # #     # < 2 * kl_original_reference.std()
    #     kl_original_reference.min() < minimum_kl_calculated < kl_original_reference.max(),
    # )
    print(
        f"{minimum_kl_calculated=}, {kl_original_reference.min()=},"
        f" {kl_original_reference.max()=}"
    )
    print(f"{kl_original_reference=}")
    if minimum_kl_calculated < kl_original_reference.min():
        pytest.skip(
            f"KL {minimum_kl_calculated} was better than all old results (Minimum:"
            f" {kl_original_reference.min()}!"
        )
    elif (
        kl_original_reference.min()
        < minimum_kl_calculated
        < kl_original_reference.max()
    ):
        pytest.skip(
            f"KL {minimum_kl_calculated} is in the expected range of reference values (Minimum:"
            f"{kl_original_reference.min()}, Maximum: {kl_original_reference.max()})"
        )
    assert minimum_kl_calculated < kl_original_reference.max(), (
        f"KL {minimum_kl_calculated} was worse than all old results! (Maximum:"
        f" {kl_original_reference.max()})"
    )
    return model.model_name, minimum_kl_transformed_iteration


def discrete_qcbm_confs():
    name = ["test_discrete_qcbm"]
    data_set_names = ["O", "X", "MG", "Stocks"]
    dimensions = [2, 3]
    circuit_depths = [3]
    normalizations = ["minmax", "pit"]
    configurations_to_validate = itertools.product(
        name, data_set_names, dimensions, circuit_depths, normalizations
    )
    return configurations_to_validate


@pytest.mark.parametrize("conf", discrete_qcbm_confs())
def test_discrete_qcbm(conf):
    print(f"{conf=}")
    name, data_set_name, dimension, circuit_depth, normalization = conf
    if dimension == 2:
        fixed_hyperparameters = {
            "n_epochs": 15000,
            "batch_size": 200,
            "hist_samples": 10000,
            "initial_sigma": 1e-2,
            "hot_starting": False,
        }
    else:
        fixed_hyperparameters = {
            "n_epochs": 20000,
            "batch_size": 200,
            "hist_samples": 10000,
            "initial_sigma": 1e-2,
            "hot_starting": False,
        }
    # fixed_hyperparameters["initial_sigma"] = 2 / circuit_depth
    if dimension == 2:
        reference_data = pd.read_csv(f"{minimum_kl_data_path}/qcbm_2D_discrete.csv")
        # Get the reference data where training_method column has value "qcbm", and integral_transform is False
        reference_data = reference_data[
            (reference_data["dataset"] == f"{data_set_name}_2D")
            & (reference_data["depth"] == circuit_depth)
            & (reference_data["training_method"] == "qcbm")
            & (reference_data["integral_transform"] == (normalization == "pit"))
        ]
    else:
        reference_data = pd.read_csv(f"{minimum_kl_data_path}/qcbm_3D_discrete.csv")
        reference_data = reference_data[
            (reference_data["dataset"] == f"{data_set_name}_3D")
            & (reference_data["depth"] == circuit_depth)
            & (reference_data["training_method"] == "qcbm")
            & (reference_data["integral_transform"] == (normalization == "pit"))
        ]

    if fixed_hyperparameters["hot_starting"]:
        for hotstart_depth in range(1, circuit_depth + 1):
            if dimension == 2:
                reference_data = pd.read_csv(
                    f"{minimum_kl_data_path}/qcbm_2D_discrete.csv"
                )
                # Get the reference data where training_method column has value "qcbm", and integral_transform is False
                reference_data = reference_data[
                    (reference_data["dataset"] == f"{data_set_name}_2D")
                    & (reference_data["depth"] == hotstart_depth)
                    & (reference_data["training_method"] == "qcbm")
                    & (reference_data["integral_transform"] == (normalization == "pit"))
                ]
            else:
                reference_data = pd.read_csv(
                    f"{minimum_kl_data_path}/qcbm_3D_discrete.csv"
                )
                reference_data = reference_data[
                    (reference_data["dataset"] == f"{data_set_name}_3D")
                    & (reference_data["depth"] == hotstart_depth)
                    & (reference_data["training_method"] == "qcbm")
                    & (reference_data["integral_transform"] == (normalization == "pit"))
                ]

            # Copy fixed_hyperparameters into a new dictionary hotstart_hyperparameters
            hotstart_hyperparameters = copy.deepcopy(fixed_hyperparameters)
            hotstart_hyperparameters["initial_sigma"] = hotstart_hyperparameters[
                "initial_sigma"
            ][hotstart_depth - 1]
            if hotstart_depth == 1:
                hot_start_path = ""
            else:
                hot_start_path = f"experiments/{previous_model_name}/parameters_training_iteration={previous_model_best_iteration}.npy"
            model = DiscreteQCBMModelHandler()
            model.build(
                name,
                data_set_name,
                n_qubits=4 * dimension,
                n_registers=dimension,
                circuit_depth=hotstart_depth,
                initial_sigma=hotstart_hyperparameters["initial_sigma"],
                circuit_type="standard" if normalization == "minmax" else "copula",
                transformation=normalization,
                hot_start_path=hot_start_path,
                slower_progress_update=True,
            )
            hot_starting_conf = (
                name,
                data_set_name,
                dimension,
                hotstart_depth,
                normalization,
            )
            (
                previous_model_name,
                previous_model_best_iteration,
            ) = helper_test_configuration(
                model, hot_starting_conf, reference_data, hotstart_hyperparameters
            )
    else:
        model = DiscreteQCBMModelHandler()
        model.build(
            name,
            data_set_name,
            n_qubits=4 * dimension,
            n_registers=dimension,
            circuit_depth=circuit_depth,
            initial_sigma=fixed_hyperparameters["initial_sigma"],
            circuit_type="standard" if normalization == "minmax" else "copula",
            transformation=normalization,
            slower_progress_update=True,
        )
        helper_test_configuration(model, conf, reference_data, fixed_hyperparameters)


def discrete_qgan_confs():
    name = ["test_discrete_qgan"]
    data_set_names = ["O", "X", "MG", "Stocks"]
    dimensions = [2, 3]
    circuit_depths = [3]
    normalizations = ["minmax", "pit"]
    configurations_to_validate = itertools.product(
        name, data_set_names, dimensions, circuit_depths, normalizations
    )
    return configurations_to_validate


@pytest.mark.parametrize("conf", discrete_qgan_confs())
def test_discrete_qgan(conf):
    print(f"{conf=}")
    name, data_set_name, dimension, circuit_depth, normalization = conf
    if dimension == 2 and data_set_name == "Stocks":
        fixed_hyperparameters = {
            "n_epochs": (
                1500 * 6
            ),  # Scale Florians value by 6 to account for the fact that "n_epochs" really means
            # "n_updates_in_total"
            "batch_size": 350,
            "learning_rate_generator": 2e-1,
            "learning_rate_discriminator": 5e-2,
        }
    elif dimension == 2:
        fixed_hyperparameters = {
            "n_epochs": 500 * 50,
            "batch_size": 1000,
            "learning_rate_generator": 2e-1,
            "learning_rate_discriminator": 5e-2,
        }
    elif dimension == 3 and data_set_name == "Stocks":
        fixed_hyperparameters = {
            "n_epochs": 3000 * 6,
            "batch_size": 350,
            "learning_rate_generator": 2e-1,
            "learning_rate_discriminator": 5e-2,
        }
    else:
        fixed_hyperparameters = {
            "n_epochs": 500 * 100,
            "batch_size": 2000,
            "learning_rate_generator": 2e-1,
            "learning_rate_discriminator": 5e-2,
        }
    model = DiscreteQGANModelHandler()

    model.build(
        name,
        data_set_name,
        n_qubits=4 * dimension,
        n_registers=dimension,
        circuit_depth=circuit_depth,
        transformation=normalization,
        circuit_type="standard" if normalization == "minmax" else "copula",
        slower_progress_update=True,
    )
    if dimension == 2:
        reference_data = pd.read_csv(f"{minimum_kl_data_path}/qgan_discrete.csv")
        # Get the reference data where training_method column has value "qcbm", and integral_transform is False
        reference_data = reference_data[
            (
                (reference_data["dataset"] == f"{data_set_name}_2d")
                | (reference_data["dataset"] == f"{data_set_name}_2D")
            )
            & (reference_data["depth"] == circuit_depth)
            & (reference_data["training_method"] == "qgan")
            & (reference_data["integral_transform"] == (normalization == "pit"))
        ]
    else:
        reference_data = pd.read_csv(f"{minimum_kl_data_path}/qgan_discrete.csv")
        reference_data = reference_data[
            (reference_data["dataset"] == f"{data_set_name}_3d")
            & (reference_data["depth"] == circuit_depth)
            & (reference_data["training_method"] == "qgan")
            & (reference_data["integral_transform"] == (normalization == "pit"))
        ]
    helper_test_configuration(model, conf, reference_data, fixed_hyperparameters)


def continuous_qcbm_confs():
    name = ["test_continuous_qcbm"]
    data_set_names = ["O", "X", "MG", "Stocks"]
    dimensions = [2, 3]
    # circuit_depths = range(1, 9)
    circuit_depths = [8]
    normalizations = ["minmax", "pit"]
    configurations_to_validate = itertools.product(
        name, data_set_names, dimensions, circuit_depths, normalizations
    )
    return configurations_to_validate


@pytest.mark.parametrize("conf", continuous_qcbm_confs())
def test_continuous_qcbm(conf):
    print(f"{conf=}")
    name, data_set_name, dimension, circuit_depth, normalization = conf
    if dimension == 2:
        fixed_hyperparameters = {
            "n_epochs": 3000,
            "batch_size": 100,
            "hist_samples": 100,
            "initial_sigma": 1e-2,
        }
    else:
        fixed_hyperparameters = {
            "n_epochs": 10000,
            "batch_size": 100,
            "hist_samples": 100,
            "initial_sigma": 1e-2,
        }
    model = ContinuousQCBMModelHandler()

    model.build(
        name,
        data_set_name,
        n_qubits=dimension,
        circuit_depth=circuit_depth,
        transformation=normalization,
        initial_sigma=fixed_hyperparameters["initial_sigma"],
        slower_progress_update=True,
    )
    if dimension == 2:
        reference_data = pd.read_csv(f"{minimum_kl_data_path}/qcbm_2D_continuous.csv")
        # Get the reference data where training_method column has value "qcbm", and integral_transform is False
        reference_data = reference_data[
            (reference_data["dataset"] == f"{data_set_name} 2D")
            & (reference_data["depth"] == circuit_depth)
            & (reference_data["training_method"] == "qcbm")
            & (reference_data["integral_transform"] == (normalization == "pit"))
        ]
    else:
        reference_data = pd.read_csv(f"{minimum_kl_data_path}/qcbm_3D_continuous.csv")
        reference_data = reference_data[
            (reference_data["dataset"] == f"{data_set_name} 3D")
            & (reference_data["depth"] == circuit_depth)
            & (reference_data["training_method"] == "qcbm")
            & (reference_data["integral_transform"] == (normalization == "pit"))
        ]
    helper_test_configuration(model, conf, reference_data, fixed_hyperparameters)


def continuous_qgan_confs():
    name = ["test_continuous_qgan"]
    data_set_names = ["O", "X", "MG", "Stocks"]
    dimensions = [2, 3]
    # circuit_depths = range(1, 9)
    circuit_depths = [8]
    normalizations = ["minmax", "pit"]
    configurations_to_validate = itertools.product(
        name, data_set_names, dimensions, circuit_depths, normalizations
    )
    return configurations_to_validate


@pytest.mark.parametrize("conf", continuous_qgan_confs())
def test_continuous_qgan(conf):
    print(f"{conf=}")
    name, data_set_name, dimension, circuit_depth, normalization = conf
    if data_set_name == "Stocks" and normalization == "pit" and dimension == 2:
        fixed_hyperparameters = {
            "n_epochs": 10000,
            "learning_rate_generator": 1e-4,
            "learning_rate_discriminator": 1e-4,
            "batch_size": None,
        }
    elif dimension == 2:
        fixed_hyperparameters = {
            "n_epochs": 10000,
            "learning_rate_generator": 1e-3,
            "learning_rate_discriminator": 1e-3,
            "batch_size": None,
        }
    elif dimension == 3 and data_set_name == "Stocks":
        fixed_hyperparameters = {
            "n_epochs": 10000,
            "learning_rate_generator": 1e-3,
            "learning_rate_discriminator": 1e-3,
            "batch_size": None,
        }
    elif dimension == 3 and data_set_name == "MG" and normalization == "minmax":
        fixed_hyperparameters = {
            "n_epochs": 10000,
            "learning_rate_generator": 1e-4,
            "learning_rate_discriminator": 1e-4,
            "batch_size": 1000,
        }
    else:
        fixed_hyperparameters = {
            "n_epochs": 10000,
            "learning_rate_generator": 1e-3,
            "learning_rate_discriminator": 1e-3,
            "batch_size": 1000,
        }
    model = ContinuousQGANModelHandler()
    print(f"{conf=}")
    print(f"{fixed_hyperparameters=}")

    model.build(
        name,
        data_set_name,
        n_qubits=dimension,
        circuit_depth=circuit_depth,
        transformation=normalization,
        slower_progress_update=True,
    )
    data_set_name_reference = (
        data_set_name if data_set_name != "MG" else "Mixed Gaussians"
    )
    if dimension == 2:
        reference_data = pd.read_csv(f"{minimum_kl_data_path}/qgan_continuous.csv")
        # Get the reference data where training_method column has value "qcbm", and integral_transform is False
        reference_data = reference_data[
            (reference_data["dataset"] == f"{data_set_name_reference} 2D")
            & (reference_data["depth"] == circuit_depth)
            & (reference_data["training_method"] == "qgan")
            & (reference_data["integral_transform"] == (normalization == "pit"))
        ]
    else:
        reference_data = pd.read_csv(f"{minimum_kl_data_path}/qgan_continuous.csv")
        reference_data = reference_data[
            (reference_data["dataset"] == f"{data_set_name_reference} 3D")
            & (reference_data["depth"] == circuit_depth)
            & (reference_data["training_method"] == "qgan")
            & (reference_data["integral_transform"] == (normalization == "pit"))
        ]
    helper_test_configuration(model, conf, reference_data, fixed_hyperparameters)


if __name__ == "__main__":
    import colorama

    print(
        f"{colorama.Fore.RED}Discrete QCBM"
        f" Configurations:{colorama.Style.RESET_ALL}\n{list(enumerate(discrete_qcbm_confs()))}\n"
    )
    print(
        f"{colorama.Fore.RED}Discrete QGAN"
        f" Configurations:{colorama.Style.RESET_ALL}\n{list(enumerate(discrete_qgan_confs()))}\n"
    )
    print(
        f"{colorama.Fore.RED}Continuous QCBM"
        f" Configurations:{colorama.Style.RESET_ALL}\n{list(enumerate(continuous_qcbm_confs()))}\n"
    )
    print(
        f"{colorama.Fore.RED}Continuous QGAN"
        f" Configurations:{colorama.Style.RESET_ALL}\n{list(enumerate(continuous_qgan_confs()))}\n"
    )
