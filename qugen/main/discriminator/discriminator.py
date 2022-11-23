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

import flax.linen as nn_jax


class Discriminator_JAX(nn_jax.Module):
    @nn_jax.compact
    def __call__(self, x):
        x = nn_jax.Dense(
            2 * x.shape[1],
            kernel_init=nn_jax.initializers.variance_scaling(
                scale=10, mode="fan_avg", distribution="uniform"
            ),
        )(x)
        x = nn_jax.leaky_relu(x)
        x = nn_jax.Dense(
            1,
            kernel_init=nn_jax.initializers.variance_scaling(
                scale=10, mode="fan_avg", distribution="uniform"
            ),
        )(x)
        x = nn_jax.leaky_relu(x)
        return nn_jax.sigmoid(x)
