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

import numpy as np


def emp_integral_trans(data: np.ndarray):
    #calling arggsort on the result of argsort creates a bijective mapping mask
    # rank contains the positions of the original elements in the sorted array
    rank = data.argsort().argsort()
    len = data.size
    ecdf = np.linspace(0, 1, len, dtype=np.float64)
    ecdf_biject = ecdf[rank]
    return ecdf_biject

#value needs to be from 0 to 1
def reverse_emp_integral_trans_single(value: float, lookup) -> float:
    value *= len(lookup) - 1

    if value.is_integer():
        return lookup[int(value)]
    else:
        valueL = np.floor(value)
        valueH = np.ceil(value)

        #linear interpolation
        return lookup[valueL]*(value-valueL) + lookup[valueH]*(valueH-value)


#same as above but using array arithmetic instead
def reverse_emp_integral_trans_np(values, lookups):
    #make sure numpy array
    values = np.array(values)
    lookups: np.ndarray = np.array(lookups)

    #assumes non ragged array
    values *= np.shape(lookups)[1]-1
    rows = np.shape(lookups)[0]

    #if we are an integer do not use linear interpolation
    valuesL = np.floor(values).astype(int)
    valuesH = np.ceil(values).astype(int)

    #if we are an integer then floor and ceiling are the same
    isIntMask = 1-(valuesH-valuesL)

    rowIndexer = np.arange(rows)
    resultL = lookups[rowIndexer, valuesL] #doing 2d lookup as [[index1.row, index2.row],[index1.column, index2.column]]
    resultH = lookups[rowIndexer, valuesH] #where 2d index tuple would be (index1.row, index1.column)

    #lookup int or do linear interpolation
    return resultL*(isIntMask + values - valuesL) +resultH * (valuesH - values)