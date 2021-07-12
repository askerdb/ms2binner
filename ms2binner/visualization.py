"""
Authors: Asker Brejnrod & Arjun Sampath

This file contains useful methods for plotting binned ms2 spectra data

Notable libraries used are:
    - numpy: https://numpy.org/doc/stable/
    - pandas: https://pandas.pydata.org
    - seaborn: https://seaborn.pydata.org
    - matplotlib: https://matplotlib.org
    - nimfa: https://nimfa.biolab.si
"""
import numpy as np
import pandas as pd
import nimfa
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

def softmax(x):
    """Compute softmax values for x

    Args:
        x: Array of values to find softmax() of

    Returns:
        Array of same dimensions as input, with the softmax of all values computed
    
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def close_windows(event):
    """Closes all Matplotlib windows
    
    Key listener function used to close all plt windows on escape
    
    Args:
        event: Keylistener event variable

    """
    if event.key == 'escape':
        plt.close('all')
    

def plot_ms2_components(binned_ms2data, num_components=10, output_file=None, headless=False):
    """ Plots binned ms2spectra data

    Takes binned ms2spectra and breaks it up into the specified number of components
    using a Non-Negative Matrix Factorization algorithm (NMF). It uses a softmax to
    normalize each component. It plots all the spectra intensities by component in
    a general graph that shows how each component relates to others by intensity

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
    
    # Use NMF to split the data into components
    nmf_model = nimfa.Nmf(binned_ms2data, rank=num_components)
    model = nmf_model()

    # Get the H matrix which is components on the rows and spectra intensities on the columns
    H = model.fit.H
    H_norm = []
    for x in H:
            H_norm.append(softmax(x.toarray()[0])) # Get the softmax of each component

    H_norm = np.array(H_norm).T # Transpose for easier use with pandas DataFrame

    labels = []
    # Create a tuple of all the component labels
    for i in np.arange(1, num_components+1):
        labels.append('Component ' + str(i))

    # Create a new DF in the proper format with component labels
    df = pd.DataFrame(H_norm, columns=labels)

    # Create a strip plot of the spectra intensities grouped by the different components
    fig = plt.figure()
    ax = sns.stripplot(data=df, size=2.5, jitter=.05)

    # Make the x-tick labels look nice/readable
    ax.set_xticklabels(ax.get_xticklabels(), rotation=55, ha='right')

    # Set y-labels and window titles. X-label is included with the DF
    ax.set_ylabel(r"$Normalized\ m/z\ Intensity$")
    ax.set_title("Spectra Intensities by Component", fontweight="bold")
    fig.canvas.set_window_title("Spectra Intensities by Component")

    plt.tight_layout()

    # If expected, create the output file and save
    if output_file != None:
        pdf_file = PdfPages(output_file)
        pdf_file.savefig()
        print("Plot saved to " + output_file)
        pdf_file.close()

    #attaches keylistener to plt figure
    fig.canvas.mpl_connect('key_press_event', close_windows)

    plt.show()

    return ax

def plot_ms2_components_fine(binned_ms2data, bins, num_components=10, output_file=None, headless=False):
    """ Plots binned ms2spectra data

    Takes binned ms2spectra and breaks it up into the specified number of components
    using a Non-Negative Matrix Factorization algorithm (NMF). It uses a softmax to
    normalize each component. It plots each individual component on a separate graph
    with bins vs intensity shown as a bar graph

    Args:
        binned_ms2data: Binned matrix of ms2spectra
        bins: Bins that correlate to the binned input matrix
        num_components: Number of components to split the spectra into
        output_file: File to save the plot to
        headless: Whether to show the plot or not for cases of plotting on a server without a GUI

    Returns:
        Tuple of  multiple Matplotlib Axes objects that contains each component's plotted graph data
        
    """
    
    if headless:
        matplotlib.use('Agg') #for plotting w/out GUI on servers
    
    # Use NMF to split the data into components
    nmf_model = nimfa.Nmf(binned_ms2data, rank=num_components)
    model = nmf_model()

    # Get the matrix that splits bins into the different components
    # Transpose so components is the rows and bins are the columns
    W = model.fit.W.T
    W_norm = []
    for x in W:
            W_norm.append(softmax(x.toarray()[0])) # calculate the softmax of each component

    subplots = []

    # If an output file is specified then create a PdfPages object for it
    pdf_file = PdfPages("fine_" + output_file) if output_file != None else None

    index = 1 # index is for determining component number

    # loop through all components
    for comp in W_norm:

        # Create new figure and plot the data
        fig = plt.figure()
        ax = sns.barplot(x=bins, y=comp*100, color='steelblue') # *100 ensures that data is as a percentage

        # Sets the locations of the x tick marks to automatic
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Labelling of axis, title, and window title
        ax.set_ylabel(r"$Normalized\ Intensity\,[\%]$")
        ax.set_xlabel(r"$Binned\ m/z$")
        ax.set_title("Component " + str(index), fontweight="bold")
        fig.canvas.set_window_title("Component " + str(index))

        # Y-axis limits, sets grid-lines parallel to x-axis and sets a tight layout
        ax.set_ylim(0,100)
        ax.grid(True, axis="y", color='black', linestyle=':', linewidth=0.1)
        plt.tight_layout()
        
        #attaches keylistener to plt figure
        fig.canvas.mpl_connect('key_press_event', close_windows)
        
        subplots.append(ax)

        # Saves the figure if expected
        if pdf_file != None and output_file != None:
            pdf_file.savefig()

        index += 1 # increment component number

    # Closes the PDF file
    if pdf_file != None and output_file != None:
        print("Plot saved to " + "file_" + output_file)
        pdf_file.close()


    plt.show()

    return subplots