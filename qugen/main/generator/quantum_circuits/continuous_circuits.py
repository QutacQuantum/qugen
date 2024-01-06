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


import jax
import jax.numpy as jnp
import pennylane as qml
import matplotlib.pyplot as plt

def get_qnode(circuit_depth, n_qubits):
    diff_method = "best"
    dev = qml.device("default.qubit", wires=n_qubits)

    num_trainable_params = 0

    def qnode_fn(inputs, weights):
        nonlocal num_trainable_params 
        num_trainable_params = 0
        for i in range(circuit_depth):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights[i], wires=range(n_qubits))
            num_trainable_params += weights[i].size

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    dummy_noise_inputs = jnp.ones((n_qubits,))
    qnode = qml.QNode(
        qnode_fn,
        device=dev,
        diff_method=diff_method,
        interface="jax"
    )
    dummy_weights = jnp.zeros((circuit_depth, 1, n_qubits, 3))
    specs = qml.specs(qnode, expansion_strategy="device")(dummy_noise_inputs, dummy_weights)
    # From the value specs["num_trainable_params"] calculated by pennylane, subtract the number of times the noise is
    # loaded into the circuit. It does not seem to be possible to specify to not count one of the function arguments.
    # Tested with both lambda and functools.partial, but it does not work. Therefore, manual subtraction is performed.
    num_trainable_params_spec = specs["num_trainable_params"] - circuit_depth * n_qubits
    qnode = jax.jit(qnode)
    #print("Trainable parameters (count): ", num_trainable_params, 
    #      "Trainable parameters (spec): ", num_trainable_params_spec)
    return qnode, num_trainable_params


if __name__ == "__main__":
    circuit_depth = 7
    n_qubits = 3
    qnode, num_params = get_qnode(circuit_depth, n_qubits)
    noise = jnp.array([0.5, -0.5])
    key = jax.random.PRNGKey(4)
    weights = jax.random.normal(key, shape=(circuit_depth, 1, n_qubits, 3)) * 2*jnp.pi - jnp.pi
    grad = jax.grad(lambda x: (qnode(noise, x)).sum())(weights)
    print(jnp.min(jnp.abs(grad)))