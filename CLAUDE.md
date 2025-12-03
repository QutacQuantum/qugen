# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qugen is a quantum generative modeling framework developed by QUTAC (Quantum Technology & Application Consortium). It implements quantum circuit-based models (QCBM) and quantum generative adversarial networks (QGAN) for both continuous and discrete data generation using PennyLane and JAX.

## Installation and Setup

### Standard Installation
```bash
conda create --name qugen python=3.9.12  # or python>=3.9
conda activate qugen
pip install -e .
```

### Known Issues and Fixes

**PennyLane/Autoray Compatibility:**
- If you see `AttributeError: module 'autoray.autoray' has no attribute 'NumpyMimic'`, downgrade autoray:
  ```bash
  pip install "autoray<0.6.0"
  ```

**Note:** As of December 2025, dependencies have been updated to address security vulnerabilities:
- setuptools>=70.0.0 (resolves Python 3.12 compatibility and CVEs)
- pytest>=8.0.0
- tqdm>=4.66.0
- colorama>=0.4.6

## Commands

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest qugen/test/generator/test_continuous_qcbm_model_handler.py

# Run specific test method
pytest qugen/test/generator/test_continuous_qcbm_model_handler.py::TestContinousQCBMModelHandler::test_train
```

### Training Models
```bash
# Run training scripts from the apps/logistics directory
cd apps/logistics
python train_continuous_qcbm.py
python train_continuous_qgan.py
python train_discrete_qcbm.py
python train_discrete_qgan.py
```

## Architecture

### Core Components

1. **Model Handlers** (`qugen/main/generator/`)
   - `BaseModelHandler`: Abstract base class defining the interface
   - `ContinuousQCBMModelHandler`: Quantum Circuit Born Machine for continuous data
   - `ContinuousQGANModelHandler`: Quantum GAN for continuous data
   - `DiscreteQCBMModelHandler`: QCBM for discrete data
   - `DiscreteQGANModelHandler`: QGAN for discrete data

2. **Quantum Circuits** (`qugen/main/generator/quantum_circuits/`)
   - Circuit implementations using PennyLane
   - Parameterized quantum circuits for generative modeling

3. **Supporting Modules**
   - `qugen/main/data/`: Data loading and transformation utilities
   - `qugen/main/discriminator/`: Classical discriminator networks
   - `qugen/main/visualization/`: Plotting and visualization tools

### Model Handler Interface

All model handlers implement these core methods:
- `build()`: Define model architecture and initialize weights
- `train()`: Train the model on provided dataset
- `predict()`: Generate samples from trained model
- `save()`: Save model weights to file
- `reload()`: Load previously saved model
- `evaluate()`: Evaluate model performance using KL divergence

**ContinuousQCBMModelHandler Additional Methods:**
- `save_circuit_qasm()`: Export quantum circuit as QASM-like string with current weights
- `save_circuit_metadata()`: Save comprehensive circuit metadata including KL divergence
- `get_circuit_qasm_string()`: Generate circuit text representation
- `get_circuit_metadata()`: Collect circuit specifications and performance metrics

**DiscreteQGANModelHandler Additional Methods:**
- `save_circuit_qasm()`: Export quantum generator circuit as QASM-like string with current weights
- `get_circuit_qasm_string()`: Generate generator circuit text representation
- Enhanced metadata: Includes circuit info and performance metrics in existing `meta.json` structure

### Training Workflow

1. **Data Loading**: Training data stored in `apps/logistics/training_data/`
2. **Model Creation**: Instantiate appropriate model handler
3. **Build**: Configure model parameters (qubits, circuit depth, etc.)
4. **Train**: Run training loop with specified epochs and batch size
5. **Evaluate**: Calculate KL divergence to assess model quality
6. **Save**: Store model artifacts in `apps/logistics/experiments/`

### File Structure
```
qugen/
├── main/
│   ├── generator/          # Model implementations
│   ├── data/              # Data utilities
│   ├── discriminator/     # Classical discriminators
│   └── visualization/     # Plotting tools
├── test/                  # Unit tests
└── apps/logistics/        # Training scripts and data
    ├── training_data/     # Input datasets
    ├── experiments/       # Model outputs
    │   └── model_name/    # Individual model directory
    │       ├── parameters_training_iteration=X.npy     # Weights (QCBM)
    │       ├── parameters_training_iteration=X.pickle  # Weights (QGAN)
    │       ├── meta.json                               # Model metadata (enhanced)
    │       ├── circuit_epoch_X.qasm                    # Circuit representation (QCBM)
    │       ├── circuit_metadata_epoch_X.json           # Circuit specs (QCBM)
    │       └── generator_circuit_epoch_X.qasm          # Generator circuit (QGAN)
    └── train_*.py         # Training scripts
```

## Development Notes

### Dependencies

**Core Framework:**
- JAX/JAXLib (0.5.3): Automatic differentiation and optimization
- PennyLane (0.42.3): Quantum computing framework
- NumPy (1.26.4): Numerical computing
- SciPy (>=1.13.0): Scientific computing

**Optimization & ML:**
- Optax (0.2.2): Gradient processing and optimization
- Flax (>=0.10.0): Neural network library
- CMA (3.2.2): Covariance Matrix Adaptation Evolution Strategy

**Development & Visualization:**
- pytest (>=8.0.0): Testing framework
- Matplotlib (>=3.5.3): Visualization
- pandas (>=1.4.3): Data manipulation

**Build Tools:**
- setuptools (>=70.0.0): Package management
- tqdm (>=4.66.0): Progress bars
- colorama (>=0.4.6): Terminal colors

### Testing Patterns
- Tests use `unittest.mock.MagicMock` to mock expensive model operations
- Each model handler has corresponding test file in `qugen/test/generator/`
- Tests verify the core interface methods work correctly

### Common Debugging
- Check PennyLane device configuration if quantum circuits fail
- Verify data shape compatibility between dataset and model qubits
- Monitor KL divergence during training for convergence issues
- Use `model.evaluate()` to find best performing weights across training epochs