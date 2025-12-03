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

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


from qugen.main.data.integral_transform import emp_integral_trans
from typing import List

# Transformation classes
class MinMaxNormalizer:
    """Min-max normalizer for quantum circuit data preprocessing.
    
    Scales data to [0,1] range using (data - min) / (max - min) formula.
    Used in QCBM/QGAN training when 'minmax' transformation is specified.
    
    Args:
        reverse_lookup: Stored min/max values for inverse transform
        epsilon: Numerical stability parameter (default: 0, typically 1e-6)
    """
    def __init__(self, reverse_lookup = None, epsilon = 0):
        self.reverse_lookup = reverse_lookup
        self.epsilon = epsilon

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.min = data.min()
        self.max = data.max() - data.min()
        data = (data - self.min) / self.max
        self.reverse_lookup = (self.min, self.max)
        return data / (1 + self.epsilon)

    def transform(self, data: np.ndarray) -> np.ndarray:
        min = data.min()
        max = data.max() - data.min()
        data = (data - min) / max
        return data / (1 + self.epsilon)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        data = data * (1 + self.epsilon)
        self.min, self.max = self.reverse_lookup
        return data * self.max + self.min


class NoTransformNormalizer:
    """Identity transform normalizer for raw data pass-through.
    
    Returns data unchanged - used when 'none' transformation is specified
    in discrete model training. Stores original data statistics for consistency.
    
    Args:
        reverse_lookup: Stored original data statistics (unused but kept for interface)
        epsilon: Numerical stability parameter (unused but kept for interface)
    """
    def __init__(self, reverse_lookup = None, epsilon = 0):
        self.reverse_lookup = reverse_lookup
        self.epsilon = epsilon

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        # Store original data statistics for inverse transform
        self.data_min = data.min(axis=0)
        self.data_max = data.max(axis=0)
        self.reverse_lookup = (self.data_min, self.data_max)
        return data

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data


class PITNormalizer():
    """Probability Integral Transform normalizer for quantum generative models.
    
    Applies empirical CDF transformation to convert data to uniform [0,1] distribution.
    Required for copula circuits and default for QCBM/QGAN training. Uses JAX for
    optimized inverse transforms.
    
    Args:
        reverse_lookup: Stored sorted values for inverse CDF lookup
        epsilon: Numerical stability parameter (default: 0, typically 1e-6)
    """
    def __init__(self, reverse_lookup = None, epsilon = 0):
        self.reverse_lookup = reverse_lookup
        self.epsilon = epsilon

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data)
        epit = df.copy(deep=True).transpose()
        reverse_epit_lookup = epit.copy(deep=True)

        epit.values[::] = [emp_integral_trans(row) for row in epit.values]
        epit = epit.transpose()
        reverse_epit_lookup.values[::] = [np.sort(row) for row in reverse_epit_lookup.values]

        df = epit.copy()
        self.reverse_lookup = reverse_epit_lookup.values
        self.reverse_lookup = jnp.array(self.reverse_lookup)
        return df.values / (1 + self.epsilon)

    def transform(self, data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data)
        epit = df.copy(deep=True).transpose()
        reverse_epit_lookup = epit.copy(deep=True)

        epit.values[::] = [emp_integral_trans(row) for row in epit.values]
        epit = epit.transpose()
        reverse_epit_lookup.values[::] = [np.sort(row) for row in reverse_epit_lookup.values]

        df = epit.copy()
        return df.values / (1 + self.epsilon)

    def _reverse_emp_integral_trans_single(self, values: jnp.ndarray) -> List[float]:
        # assumes non ragged array
        values = values * (jnp.shape(self.reverse_lookup)[1] - 1)
        rows = jnp.shape(self.reverse_lookup)[0]
        # if we are an integer do not use linear interpolation
        valuesL = jnp.floor(values).astype(int)
        valuesH = jnp.ceil(values).astype(int)
        # if we are an integer then floor and ceiling are the same
        isIntMask = 1 - (valuesH - valuesL)
        rowIndexer = jnp.arange(rows)
        resultL = self.reverse_lookup[([rowIndexer], [valuesL])]  # doing 2d lookup as [[index1.row, index2.row],[index1.column, index2.column]]
        resultH = self.reverse_lookup[([rowIndexer], [valuesH])]  # where 2d index tuple would be (index1.row, index1.column)
        # lookup int or do linear interpolation
        return resultL * (isIntMask + values - valuesL) + resultH * (valuesH - values)    

    @partial(jax.jit, static_argnums=(0,))
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        data = data * (1 + self.epsilon)
        res = jax.vmap(self._reverse_emp_integral_trans_single)(data)
        # res = [self._reverse_emp_integral_trans_single(row) for row in data]
        return res[:, 0, :]


class PITNormalizerNumPy:
    """NumPy-only version of PITNormalizer for JAX-free environments.
    
    Provides identical probability integral transform functionality as PITNormalizer
    but uses pure NumPy instead of JAX. Use when JAX compatibility issues arise
    or when JAX optimization is not needed.
    
    Args:
        reverse_lookup: Stored sorted values for inverse CDF lookup
        epsilon: Numerical stability parameter (default: 0, typically 1e-6)
    """
    def __init__(self, reverse_lookup=None, epsilon=0):
        self.reverse_lookup = reverse_lookup
        self.epsilon = epsilon

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data)
        epit = df.copy(deep=True).transpose()
        reverse_epit_lookup = epit.copy(deep=True)

        epit.values[::] = [emp_integral_trans(row) for row in epit.values]
        epit = epit.transpose()
        reverse_epit_lookup.values[::] = [np.sort(row) for row in reverse_epit_lookup.values]

        df = epit.copy()
        self.reverse_lookup = reverse_epit_lookup.values
        self.reverse_lookup = np.array(self.reverse_lookup)
        return df.values / (1 + self.epsilon)

    def transform(self, data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data)
        epit = df.copy(deep=True).transpose()
        reverse_epit_lookup = epit.copy(deep=True)

        epit.values[::] = [emp_integral_trans(row) for row in epit.values]
        epit = epit.transpose()
        reverse_epit_lookup.values[::] = [np.sort(row) for row in reverse_epit_lookup.values]

        df = epit.copy()
        return df.values / (1 + self.epsilon)

    def _reverse_emp_integral_trans_single(self, values: np.ndarray) -> np.ndarray:
        values = values * (np.shape(self.reverse_lookup)[1] - 1)
        rows = np.shape(self.reverse_lookup)[0]
        
        valuesL = np.floor(values).astype(int)
        valuesH = np.ceil(values).astype(int)
        
        isIntMask = 1 - (valuesH - valuesL)
        rowIndexer = np.arange(rows)
        
        resultL = self.reverse_lookup[rowIndexer, valuesL]
        resultH = self.reverse_lookup[rowIndexer, valuesH]
        
        return resultL * (isIntMask + values - valuesL) + resultH * (valuesH - values)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        data = data * (1 + self.epsilon)
        
        result = []
        for row in data:
            transformed_row = self._reverse_emp_integral_trans_single(row)
            result.append(transformed_row)
        
        return np.array(result)


def load_data(data_set, n_train=None, n_test=None):
    """Load training data from .npy files for quantum model training.
    
    Loads data from the logistics training_data directory for QCBM/QGAN models.
    
    Args:
        data_set: Path to .npy file (without extension)
        n_train: Maximum number of training samples to load
        n_test: Unused parameter (kept for compatibility)
        
    Returns:
        tuple: (training_data, empty_list) - training data as numpy array
    """
    train = np.load(data_set + '.npy')
    if n_train is not None:
        train = train[:n_train]
    return train, []


