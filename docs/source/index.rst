.. qugen documentation master file, created by
   sphinx-quickstart on Tue Sep 26 10:43:50 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qugen's documentation!
=================================

qugen is a module for creating and training quantum generative machine learning models on simulated quantum hardware. It is implemented using pennylane with jax acceleration.


This module accompanies the publication  `A characterization of quantum generative models <https://arxiv.org/abs/2301.09363>`_ , which gives a background on these models as well as describing in detail their structure and performance. 
The results given in this paper can be replicated with the code shared here. The authors hope it may also be of interest to the quantum machine learning community as an example of how such models can be implimented. 


We have implemented 2 different types of variational quantum ciruits:

**Discrete**: in which each measurement of the output of the circuit is interpreted directly as a discrete bit string.

**Continuous**: in which the outputs are expectation values which are interpreted as continuous valued samples
Together with these circuits, we have also implemented 2 different training methods:

**QCBM**: in which a histogram of the distribution of the training data is learned directly by the variational quantum circuit.
**QGAN**: in which samples of the data and generated data are passed to a classical discriminator neural network that judges the validity of the samples.


There are 8 training data sets, 4 2D and 4 3D each:  a circle, a cross, a mixed Gaussian data set and stocks data. 
The package allows models to be created, trained and evaluated.


Contents
--------

.. toctree::
   readme_link
   base_model
   discrete_qcbm
   cont_qcbm
   discrete_qgan
   cont_qgan
   credit_license
   
 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
