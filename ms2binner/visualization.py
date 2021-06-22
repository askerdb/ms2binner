"""
Authors: Asker Brejnrod & Arjun Sampath

This file contains useful methods for plotting binned ms2 spectra data

Notable libraries used are:
    - numpy: https://numpy.org/doc/stable/
    - pandas: https://pandas.pydata.org
    - seaborn: https://seaborn.pydata.org
    - matplotlib: https://matplotlib.org 
"""

import bin
import numpy as np
import pandas as pd
import nimfa
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def savePlot(filename, filetype='png'):
    """ Saves plot as to the inputted file
    
    Args:
    filename: File to save plot to
    filetype: File type to save plot as (default png)
    """
    plt.savefig(filename + "." + filetype, dpi=300, format=filetype)
    print("Plot saved to " + filename + "." + filetype)

def close_windows(event):
    """Key listener function used to close all plt windows on escape"""
    if event.key == 'escape':
        plt.close('all')

def plot_ms2data(binned_ms2data, num_components=10, output_file=None, headless=False):
    """ Plots binned ms2spectra data

    Takes binned ms2spectra and breaks it up into the specified number of components
    using a Non-Negative Matrix Factorization algorithm (NMF). It uses a softmax to
    normalize each component, and plots all the spectra intensities by component

    Args:
    binned_ms2data: Binned matrix of ms2spectra
    num_components: Number of components to split the spectra into
    output_file: File to save the plot to
    headless: Whether to show the plot or not for cases of plotting on a server without a GUI

    Returns:
    Matplotlib Axes object that contains the plotted graph data
    """
    if headless:
        matplotlib.use('Agg') #for plotting w/out GUI on servers
    
    nmf_model = nimfa.Nmf(binned_ms2data, rank=num_components)
    model = nmf_model()

    H = model.fit.H
    H_norm = []
    for x in H:
            H_norm.append(softmax(x.toarray()[0]))

    H_norm = np.array(H_norm).T

    labels = []
    for i in np.arange(1, num_components+1):
        labels.append('Component ' + str(i))

    df = pd.DataFrame(H_norm, columns=labels)
    ax = sns.stripplot(data=df, size=2.5, jitter=.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=55, ha='right')
    ax.set_ylabel("Normalized M/Z Intesity")
    plt.tight_layout()

    if output_file != None:
        savePlot(output_file)

    plt.gcf().canvas.mpl_connect('key_press_event', close_windows) #attaches keylistener to plt figure

    plt.show()

    return ax