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

from qugen.main.generator.continuous_qgan_model_handler import (
    ContinuousQGANModelHandler,
)
from qugen.main.data.data_handler import load_data


data_set_name = "O_3D"
data_set_path = f"./training_data/{data_set_name}"
data, _ = load_data(data_set_path)
model = ContinuousQGANModelHandler()

# build a new model:

model.build(
    model_name= "continuous",
    data_set=data_set_name, 
    n_qubits=data.shape[1], 
    circuit_depth=8, 
    transformation='pit'
)

# train a quantum generative model:

model.train(
    data, 
    n_epochs=1000, 
    initial_learning_rate_generator=1e-3, 
    initial_learning_rate_discriminator=1e-3, 
    batch_size=1000
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

# # load existing model:

# model_name  = 'continuous_X_2D_copula_pit_qgan_d320'  # example model name
# new_model = ContinuousQGANModelHandlerJAX().reload(model_name, epoch=500)

# # re-train model from pre-trained existing model

# new_model.train(data[0],
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
