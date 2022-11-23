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
import pandas as pd 
import matplotlib.pyplot as plt

def plot_kl_against_epoch(log_file_path, meta_file_path):
    """Function to take a log file from a model training and plot the KL
     against epoch up to that point in the training"""

    logs  = np.load(log_file_path, allow_pickle=True)
    logs = pd.DataFrame(logs[1:],columns=logs[0])
    logs = logs.apply(pd.to_numeric)
    plt.xlabel('iternations')
    plt.ylabel('KL in transformed space')
    plt.ylim(bottom=0)
    plt.plot(logs.iteration, logs.kl_div_transformed)
    return logs 

def scatter_plot(
    data: np.ndarray,
    title: str,
    size: tuple = (5, 4),
    x_label: str = "x",
    y_label: str ="y",
    xy_limit: list = None):
    """Build and display scatter plot from 2D data.

    Args:
        data: Two-dimensional data array
        title: Plot title
        size: (Optional) Plot size by tuple
        x_label: (Optional) Label for x-axis
        y_label: (Optional) Label for y-axis
        xy_limit: (Optional) Display limit for x/y axis
    """
    # Sanity check: data dimension
    if data.ndim != 2:
        raise ValueError("Data array with 2 dimensions not provided")

    # Create plot
    plt.rcParams["figure.figsize"] = size
    plt.scatter(data[:, 0], data[:, 1], s=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    if xy_limit:
        plt.xlim(xy_limit)
        plt.ylim(xy_limit)
    plt.show()


def hist_marginal_plot(
    data: np.ndarray,
    number_bins: int,
    size: tuple = (8, 8),
    x_range: list = None,
    y_range: list = None,
    left_margin: float = 0.1,
    bottom_margin: float = 0.1,
    width: float = 0.65,
    height: float = 0.65,
    spacing: float = 0.04):
    """Build and return figure of 2-D histogram
       with marginal histogram on the x/y axis.

    Recipe:
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html

    Args:
        data: Two-dimensional data array
        number_bins: Number of histogram bins
        size: (Optional) Plot size by tuple
        x_range: (Optional) Expected range in x
        y_range: (Optional) Expected range in y
        left_margin: (Optional) Margin on left side
        bottom_margin: (Optional) Margin on bottom side
        width: (Optional) Width of plotting area
        height: (Optional) Height of plotting area
        spacing: (Optional) spacing
    """
    # Sanity check: data dimension
    if data.ndim != 2:
        raise ValueError("Data array with 2 dimensions not provided")

    # Settings
    rect_scatter = [left_margin, bottom_margin, width, height]
    rect_histx = [left_margin, bottom_margin + height + spacing, width, 0.2]
    rect_histy = [left_margin + width + spacing, bottom_margin, 0.2, height]

    # X/Y ranges
    if not x_range:
        x_range = [np.amin(data[:, 0]), np.amax(data[:, 0])]
    if not y_range:
        y_range = [np.amin(data[:, 1]), np.amax(data[:, 1])]

    # Create square figure
    fig = plt.figure(figsize=size)

    # Add axes
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # No labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # Create 2D histogram
    # pylint: disable=line-too-long
    ax.hist2d(data[:, 0], data[:, 1], bins=(number_bins, number_bins), range=[x_range, y_range])
    ax_histx.hist(data[:, 0], bins=number_bins, range=x_range, density=False)
    ax_histy.hist(data[:, 1], bins=number_bins, range=y_range, orientation="horizontal", density=False)
    # pylint: enable=line-too-long

    return fig


