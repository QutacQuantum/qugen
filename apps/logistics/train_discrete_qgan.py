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

from qugen.main.generator.discrete_qgan_model_handler import DiscreteQGANModelHandler
from qugen.main.data.data_handler import load_data

data_set_name = "MG_2D"
data_set_path =  f"./training_data/{data_set_name}"
data, _ = load_data(data_set_path)
model = DiscreteQGANModelHandler()

# build a new model:

print('start')
model.build(model_name='discrete', 
            data_set_name=data_set_name, 
            n_qubits=8, 
            n_registers=2, 
            circuit_depth=3,
            transformation='pit',
            circuit_type='copula')

# train a quantum generative model:

model.train(
    data, 
    n_epochs= 1500 * 6, 
    initial_learning_rate_generator=2e-1, 
    initial_learning_rate_discriminator=5e-2,
    batch_size=350,
)

# evaluate the performance of the trained model:

evaluation_df = model.evaluate(data)

# find the model with the minimum Kullbach-Liebler divergence:

minimum_kl_data = evaluation_df.loc[evaluation_df["kl_original_space"].idxmin()]
minimum_kl_calculated = minimum_kl_data["kl_original_space"]
print(f"{minimum_kl_calculated=}")

# # --------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------
# # ADDITIONAL FUNCTIONALITY
# # Uncomment the following section if you want to train a model for already pre-trained weights

# load existing model:

# model_name  = 'discrete_O_2D_copula_pit_qgan'  # example model name
# new_model = DiscreteQGANModelHandler().reload(model_name, epoch=500)

# # re-train model from pre-trained existing model

# new_model.train(data,
#             n_epochs = 25,
#             batch_size=200,
#             hist_samples = 10000,
#             )

# # generate samples from a trained model:

# number_samples = 10000
# samples = new_model.predict(number_samples)

# # plot 2D samples:

# import matplotlib.pyplot as plt
# plt.scatter(samples[:, 0], samples[:, 1])
# plt.savefig('generated_samples.png')
