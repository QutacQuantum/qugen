 ## Installation

1) Create a virtual environment, e.g. using ``conda create --name qugen_env python=3.9.12``. Python 3.9 or later is supported.
2) Activate the enviroment, e.g ``source activate qugen_env``.
3) Run ``pip install .`` or ``pip install -e .`` to install it in editable mode.

Note JaxLib 0.3.25 as specified in `setup.py` might not be available via PyPi and must be installed manually:

    pip install jaxlib==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html


## Instructions for training and running models
Each model type has an example script to create and train a model found  at ''apps/logistics/train_xyz.py'' file, e.g. ''apps/logistics/train_continuous_qgan.py''

The training data, data transformation and other hyperparameters can all be specified in these training scripts.

Each model type also has a model handler, they are found in qugen/main/generator and are imported by each training script as needed.

The model handlers contain four important methods, shown in the table below.

|Method | Description |
|---|---|
|build| Builds a model by specifying its hyperparameters.|
|train| Trains a model using a training data set.|
|predict| Samples from a trained model.|
|reload| Reloads a trained model from file.|

To train a model, the training scripts work as follows:

- Each train file loads an input training data set from file (found in training_data/) and creates the appropriate model handler.
- The model handler builds a model from the specified parameters: model_name, data_set_name, n_qubits, n_registers, circuit_depth, n_epochs etc.
- A meta json file of the model is saved to experiments/modelname.
- The built model is then trained using the training data set.
- During training, at each epoch, the  weights and a log file (containing the KL divergence) are saved to experiments/modelname.

To load a model from file (see train_discrete_copula_qcbm for example):
 - Create an instance of the appropriate model handler (e.g. ``model = ContinuousQGANModelHandler()``).
 - Load the model using the reload functions and the meta json and your chosen parameters file.

To sample from a trained model (see train_discrete_copula_qcbm for example):
- Use the model.predict method, which takes as an argument the number of samples to be generated.

To evaluate a trained model:
- Use the evaluator method which takes all the weights from the training run of the model and finds the set of weights with the lowest cost function (KL divergence)



## Citation


```
@article{riofrio2023performance,
  title={A performance characterization of quantum generative models},
  author={Riofr{\'\i}o, Carlos A and Mitevski, Oliver and Jones, Caitlin and Krellner, Florian and Vu{\v{c}}kovi{\'c}, Aleksandar and Doetsch, Joseph and Klepsch, Johannes and Ehmer, Thomas and Luckow, Andre},
  journal={arXiv preprint arXiv:2301.09363},
  year={2023}
}
```

## Contact

info@qutac.de

