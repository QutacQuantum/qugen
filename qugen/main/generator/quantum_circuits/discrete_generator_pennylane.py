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

import pennylane as qml
import jax
import jax.numpy as jnp

from functools import partial
from itertools import combinations

# Since we're using a qnode with sampling, we cannot use autograd. Instead, use this (analogously to the old implementation)
@partial(jax.jit, static_argnames=["discriminator"])
def compute_gradient_JAX(samples, discriminator, discriminator_weights):
    def criterion(outputs):
        return (-1.0 * jnp.log(outputs) ).mean()

    gradient = []
    for i in range(0, len(samples), 2):
        forward_fake = samples[i]
        backward_fake = samples[i + 1]

        forward_output = discriminator.apply(discriminator_weights, forward_fake).flatten()
        backward_output = discriminator.apply(discriminator_weights, backward_fake).flatten()

        forward_diff = criterion(forward_output)
        backward_diff = criterion(backward_output)
        gradient.append(1 / 2 * (forward_diff - backward_diff))

    return jnp.array(gradient)

def discrete_copula_circuit_JAX(n_qubits, n_registers, circuit_depth):
    number_trainable_parameters = 0
    def copula_block(n_qubits, n_registers):
        n = n_qubits // n_registers
        for i in range(n):
            qml.Hadamard(wires=i)

        for j in range(n_registers - 1):
            for k in range(n):
                qml.CNOT(wires=[k, k + n * (j + 1)])

    def copula_parametric(weights, n_qubits, n_registers, circuit_depth):
        n = n_qubits // n_registers

        nonlocal number_trainable_parameters
        number_trainable_parameters = 0
        count = 0
        for _ in range(circuit_depth):
            for k in range(n):
                for j in range(n_registers):
                    qml.RZ(weights[count], wires=j * n + k)
                    qml.RX(weights[count + 1], wires=j * n + k)
                    qml.RZ(weights[count + 2], wires=j * n + k)
                    count += 3
            for i, j in combinations(range(n), 2):
                for l in range(n_registers):
                    qml.IsingXX(weights[count], wires=[l * n + i, l * n + j])
                    count += 1
        
        number_trainable_parameters = count

    def qnode_fn(weights):
        copula_block(n_qubits, n_registers=n_registers)
        copula_parametric(weights, n_qubits, n_registers=n_registers,
                          circuit_depth=circuit_depth)
        return qml.sample()
    # Only the circuit with one copula block is implemented for now.
    @partial(jax.jit, static_argnames=["n_shots"])
    def qnode_with_variable_random_key(key, weights, n_shots):
        dev = qml.device("default.qubit.jax", prng_key=key, wires=n_qubits, shots=n_shots)
        qnode = qml.QNode(qnode_fn, dev, diff_method=None, interface="jax")
        return qnode(weights)

    # Create a dummy device and a dummy qnode since we need to specify the random key and the number of shots
    # in order to create a device, then the qnode and then run the circuit to calculate the number of parameters by tracing
    # through the computation. We need to do this as what we need inside the actual model handlers is a function
    # with variable random key and shots (qnode_with_variable_random_key_and_shots) which internally creates the device,
    # qnode and runs it, meaning that no QNode object actually exists at this point in the code. Both the dummy_qnode and
    # the qnode_with_variable_random_key_and_shots function actually use the same qnode_fn though (which specifies the
    # circuit)
    dummy_device = qml.device("default.qubit.jax", prng_key=jax.random.PRNGKey(1), wires=n_qubits, shots=1)
    dummy_qnode = qml.QNode(qnode_fn, dummy_device, diff_method=None, interface="jax")
    # Need to pass in a dummy array jnp.zeros((1,)) to get the number of trainable parameters since in some cases this
    # number actually depends on the input array (because it is inferred from it), e.g. in templates like
    # qml.specs(dummy_qnode)(jnp.zeros((1,)))["num_trainable_params"]
    
    # execute node once to get number of trainable parameters
    dummy_qnode(jnp.zeros((1,)))
    return qnode_with_variable_random_key, number_trainable_parameters


def discrete_standard_circuit_JAX(n_qubits, n_registers, circuit_depth):
    
    number_trainable_parameters = 0

    def standard_parametric(weights, n_qubits, n_registers, circuit_depth):
        nonlocal number_trainable_parameters
        number_trainable_parameters = 0
        count = 0
        for _ in range(circuit_depth):
            for k in range(n_qubits):
                qml.RY(weights[count], wires=k)
                count += 1
            for k in range(n_qubits-1):
                qubit_1 = k
                qubit_2 = k + 1
                qml.IsingYY(weights[count], wires=[qubit_1, qubit_2])
                count += 1

            for k in range(n_qubits-1):
                control_qubit = k
                target_qubit = k+1
                qml.CRY(weights[count], wires=[control_qubit, target_qubit])
                count += 1
        number_trainable_parameters = count

    def qnode_fn(weights):
        standard_parametric(weights, n_qubits, n_registers=n_registers,
                            circuit_depth=circuit_depth)
        return qml.sample()

    @partial(jax.jit, static_argnames=["n_shots"])
    def qnode_with_variable_random_key_and_shots(key, weights, n_shots):
        # Only the circuit with one copula block is implemented for now.
        dev = qml.device("default.qubit.jax", prng_key=key, wires=n_qubits, shots=n_shots)
        qnode = qml.QNode(qnode_fn, dev, diff_method=None, interface="jax")
        return qnode(weights)

    # Create a dummy device and a dummy qnode since we need to specify the random key and the number of shots
    # in order to create a device, then the qnode and then run the circuit to calculate the number of parameters by tracing
    # through the computation. We need to do this as what we need inside the actual model handlers is a function
    # with variable random key and shots (qnode_with_variable_random_key_and_shots) which internally creates the device,
    # qnode and runs it, meaning that no QNode object actually exists at this point in the code. Both the dummy_qnode and
    # the qnode_with_variable_random_key_and_shots function actually use the same qnode_fn though (which specifies the
    # circuit)
    dummy_device = qml.device("default.qubit.jax", prng_key=jax.random.PRNGKey(1), wires=n_qubits, shots=1)
    dummy_qnode = qml.QNode(qnode_fn, dummy_device, diff_method=None, interface="jax")
    # Need to pass in a dummy array jnp.zeros((1,)) to get the number of trainable parameters since in some cases this
    # number actually depends on the input array (because it is inferred from it), e.g. in templates like
    # qml.specs(dummy_qnode)(jnp.zeros((1,)))["num_trainable_params"]
    
    # execute node once to get number of trainable parameters
    dummy_qnode(jnp.zeros((1,)))
    return qnode_with_variable_random_key_and_shots, number_trainable_parameters


def center(coord, n):
    return jnp.array(coord) / n + 0.5 / n


def generate_samples(key, binary_samples, n_registers, n_qubits, noisy=True):
    width = 1/(jnp.power(2**n_qubits, (1/n_registers)))
    noise = 0.5*width*jax.random.uniform(
        key,
        minval=-1, maxval=1,
        shape=(len(binary_samples), n_registers)) if noisy else jnp.zeros((len(binary_samples), n_registers))
    noise = jnp.array(noise)

    n = 2**(n_qubits//n_registers)
    # Split the binary strings for each dimension into separate arrays
    samples_dims = []
    for dim in range(n_registers):
        samples_dims.append(binary_samples[:,dim*n_qubits//n_registers : (dim + 1)*n_qubits//n_registers])
        
    # Calculate the decimal representation of the binary strings
    indices_fn = jax.vmap(lambda sample: jnp.dot(2 ** jnp.arange(0, sample.size), sample[::-1]))
    points = jnp.vstack([center(indices_fn(dim), n) for dim in samples_dims]).T 

    return points + noise