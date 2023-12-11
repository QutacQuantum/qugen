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

from setuptools import setup, find_packages
import platform

if platform.system() == 'Linux':
    setup(name='qugen',
          version='0.1',
          author='Quantum Technology & Application Consortium (QUTAC)',
          license='Apache License 2.0',
          packages=find_packages(),
          install_requires=[
              'cma==3.2.2',
              'colorama==0.4.5',
              'flax==0.6.0',
              'jax==0.3.25',
              'jaxlib==0.3.25',
              'matplotlib==3.5.3',
              'numpy==1.23.2',
              'optax==0.1.3',
              'pandas==1.4.3',
              'PennyLane==0.29.0',
              'pytest==7.4.0',
              'scipy==1.11.3',
              'setuptools==61.2.0',
              'tqdm==4.64.1', ])
elif platform.system() == 'Darwin':
    setup(name='qugen',
          version='0.1',
          author='Quantum Technology & Application Consortium (QUTAC)',
          license='Apache License 2.0',
          packages=find_packages(),
          install_requires=[
              'cma==3.2.2',
              'colorama==0.4.5',
              'flax==0.6.0',
              'jax==0.3.25',
              'jaxlib==0.3.25',
              'matplotlib==3.5.3',
              'numpy==1.23.2',
              'optax==0.1.3',
              'pandas==1.4.3',
              'PennyLane==0.29.0',
              'pytest==7.4.0',
              'scipy==1.11.3',
              'setuptools==61.2.0',
              'tqdm==4.64.1', ])
elif platform.system() == 'Windows':
    setup(name='qugen',
          version='0.1',
          author='Quantum Technology & Application Consortium (QUTAC)',
          license='Apache License 2.0',
          packages=find_packages(),
          install_requires=[
              'cma==3.2.2',
              'colorama==0.4.5',
              'flax==0.6.0',
              'jax==0.4.13',
              'jaxlib==0.4.13',
              'matplotlib==3.5.3',
              'numpy==1.23.2',
              'optax==0.1.3',
              'pandas==1.4.3',
              'PennyLane==0.29.0',
              'pytest==7.4.0',
              'scipy==1.9.0',
              'setuptools==61.2.0',
              'tqdm==4.64.1', ])
else:
    raise OSError('Unknown Operating System: {} {}'.format(platform.os.name, platform.system()))

