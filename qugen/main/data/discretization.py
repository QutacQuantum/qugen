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

from itertools import product
import numpy as np


def center_2d(i, j, n):
    ax, bx = 0., 1.
    ay, by = 0., 1.
    return ax + (2 * i + 1) / (2 * n) * (bx - ax), ay + (2 * j + 1) / (2 * n) * (by - ay)


def center(coord, n):
    return np.array(coord) / n + 0.5 / n


def compute_discretization(n_qubits, n_registered):
    format_string = "{:0" + str(n_qubits) + "b}"
    n = 2 ** (n_qubits // n_registered)
    dict_bins = {}
    for k, coordinates in enumerate(product(range(n), repeat=n_registered)):
        dict_bins.update({
            format_string.format(k): [coordinates, center(coordinates, n)]
        })
    return dict_bins
