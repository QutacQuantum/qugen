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
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Union, Sequence

def random_angle(n):
    return np.random.rand(n) * np.pi


def kl_divergence(p, q):
    eps = 1e-6
    cost = jnp.sum( p * jnp.log((p+eps)/(q+eps)) )
    return cost


def discretized_2d_probability_distribution(data, n_bins):
    x_min, x_max = np.min(data[:,0]), np.max(data[:,0])
    y_min, y_max = np.min(data[:,1]), np.max(data[:,1])
    return np.histogram2d(data[:,0], data[:,1], bins=(n_bins,n_bins), range=[[x_min, x_max], [y_min, y_max]], normed=None, weights=None, density=None)[0]/data.shape[0]


def kl_divergence_from_data(
    training_data: np.ndarray,
    learned_data: np.ndarray,
    number_bins: int = 16, 
    bin_range: Union[Sequence[Union[float, int]], Sequence[Sequence[Union[float, int]]], None] = None,
    dimension: int = 2,
):
    """
    Calculate the KL divergence, given training and learned/generated data. 
    By default, this function expects 2D data, but this can be changed using the argument "dimension". 

    Args:
        training_data (np.ndarray): The training data with shape (num_samples, num_dimensions).
        learned_data (np.ndarray): The learned data with shape (num_samples, num_dimensions).
        number_bins (int): The number of bins per dimension, i.e. the total number of D-dimensional bins is number_bins**dimension.
        bin_range Sequence[Union[float, int]] or Sequence[Sequence[Union[float, int]]]: The bin range, either specified for all axis with a single sequence or a sequence of bin-ranges for each individual dimension.
                  By default, the bin_range in each dimension is calculated from the min/max of the training_data.
        dimension (int): The dimensionality of the dataset.

    Returns:
        float: The KL-divergence.

    """
    training_data = training_data[:, :dimension]
    learned_data = learned_data[:, :dimension]
    if bin_range is None:
        b_ranges = [(training_data[:, i].min(), training_data[:, i].max()) for i in range(dimension)]
    elif isinstance(bin_range[0], int) or isinstance(bin_range[0], float):
        b_ranges = [bin_range for _ in range(dimension)]
    else:
        b_ranges = bin_range
    trained_histogram_np = np.histogramdd(training_data, bins=number_bins, range=b_ranges)
    learned_histogram_np = np.histogramdd(learned_data, bins=number_bins, range=b_ranges)
    train_probability = trained_histogram_np[0]/np.sum(trained_histogram_np[0])
    learned_probability = learned_histogram_np[0]/np.sum(learned_histogram_np[0])
    return kl_divergence(train_probability, learned_probability)


def kl_divergence_from_data_3d(training_data: np.ndarray, learned_data: np.ndarray, number_bins=16, bin_range=[[0, 1], [0, 1], [0, 1]]):
    trained_histogram_np = np.histogramdd(training_data, bins=(number_bins, number_bins, number_bins), range=bin_range)
    learned_histogram_np = np.histogramdd(learned_data, bins=(number_bins, number_bins, number_bins), range=bin_range)
    #trained_histogram = plt.hist2d(training_data[:, 0], training_data[:, 1], bins=(number_of_bins, number_of_bins), range=[[0, 1], [0, 1]])
    #learned_histogram = plt.hist2d(learned_data[:, 0], learned_data[:, 1], bins=(number_of_bins, number_of_bins), range=[[0, 1], [0, 1]])
    train_probability = trained_histogram_np[0]/np.sum(trained_histogram_np[0])
    learned_probability = learned_histogram_np[0]/np.sum(learned_histogram_np[0])
    return kl_divergence(train_probability, learned_probability)


# Convenient plotting
def plot_samples(data, title, size=(5, 4), x_label='x', y_label='y', constrained=True):
    plt.rcParams["figure.figsize"] = size
    plt.scatter(data[:, 0], data[:, 1], s=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    if constrained:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    plt.show()


def create_histogram_marginal_plot(data, number_bins):
    """ Create 2-D histogram with marginal histogram on the x/y axix
        Recipe: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    """
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.04

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # 2d histogram
    ax.hist2d(data[:, 0], data[:, 1], bins=(number_bins, number_bins), range=[[0, 1], [0, 1]])
    ax_histx.hist(data[:, 0], bins=number_bins, range=[0, 1], density=False)
    ax_histy.hist(data[:, 1], bins=number_bins, range=[0, 1], orientation='horizontal', density=False)
    return fig


class CustomDataset:
    def __init__(self,data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass


    @property
    def data(self):
        return self._data

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]