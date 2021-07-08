"""
Authors: Asker Brejnrod & Arjun Sampath

This file contains useful methods for plotting binned ms2 spectra data

Notable libraries used are:
    - numpy: https://numpy.org/doc/stable/
    - pandas: https://pandas.pydata.org
    - seaborn: https://seaborn.pydata.org
    - matplotlib: https://matplotlib.org 
"""
import numpy as np
import pandas as pd
import nimfa
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def close_windows(event):
    """Key listener function used to close all plt windows on escape"""
    if event.key == 'escape':
        plt.close('all')

def plot_ms2_components(binned_ms2data, num_components=10, output_file=None, headless=False):
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
    ax.set_ylabel(r"$Normalized m/z Intesity$")
    plt.tight_layout()

    if output_file != None:
        pdf_file = PdfPages(output_file)
        pdf_file.savefig()
        print("Plot saved to " + output_file)
        pdf_file.close()

    plt.gcf().canvas.mpl_connect('key_press_event', close_windows) #attaches keylistener to plt figure

    plt.show()

    return ax

def plot_ms2_histograms(binned_ms2data, bins, num_components=10, output_file=None, headless=False):
    if headless:
        matplotlib.use('Agg') #for plotting w/out GUI on servers
    
    nmf_model = nimfa.Nmf(binned_ms2data, rank=num_components)
    model = nmf_model()

    W = model.fit.W
    W_norm = []
    for x in W:
            W_norm.append(softmax(x.toarray()[0]))

    W_norm = np.array(W_norm).T

    subplots = []

    pdf_file = PdfPages(output_file) if output_file != None else None

    for comp in W_norm:
        plt.figure()
        ax = sns.barplot(x=bins, y=comp*100)
        ax.set_ylabel(r"$Normalized Intesity [%]$")
        ax.set_xlabel(r"$Binned m/z$")
        for ind, label in enumerate(ax.get_xticklabels()):
            if ind % 10 == 0:  # every 10th label is kept
                label.set_visible(True)
            else:
                label.set_visible(False)
        ax.set_ylim(0,100)
        plt.tight_layout()
        subplots.append(ax)

        if pdf_file != None and output_file != None:
            pdf_file.savefig()


    if pdf_file != None and output_file != None:
        print("Plot saved to " + output_file)
        pdf_file.close()

    plt.gcf().canvas.mpl_connect('key_press_event', close_windows) #attaches keylistener to plt figure

    plt.show()

    return subplots