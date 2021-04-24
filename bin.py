from pyteomics import mgf, mzxml
import numpy as np
from scipy.sparse import dok_matrix
import math
import time
import pickle as pkl
import os

def filter_zero_cols(csr):
    """ Removes all columns that only contain zeroes

    Args:
    csr: Input CSR matrix to filter

    Returns:
    A sparse CSR matrix with zero-sum columns filtered out and a boolean array 
    indicating whether to "keep" each column
    """
    # Sums each column and creates a boolean array of whether each 
    # summed column is greater than 0
    keep = np.array(csr.sum(axis = 0) > 0).flatten()
    # Only keeps columns that have a corresponding True value in keep
    csr = csr[:,keep]

    return(csr, keep)

def filter_zero_rows(csr):
    """ Removes all rows that only contain zeroes

    Args:
    csr: Input CSR matrix to filter

    Returns:
    A sparse CSR matrix that has all zero-sum rows filtered out and a boolean
    array indicating whether to "keep" each row
    """
    # Sums each row and creates a boolean array of whether each 
    # summed row is greater than 0
    keep = np.array(csr.sum(axis = 1) > 0).flatten()
    # Only keeps rows that have a corresponding True value in keep
    csr = csr[keep]

    return(csr, keep)

def row_filter_intensity(X, bin_names, threshold = 1/100):
    colsums = np.array(X.sum(axis = 0)).flatten()
    for i in range(X.shape[1]):
        X[:, i] = X[:, i]/colsums[i]
    rowsums = np.array(X.sum(axis = 1)).flatten()
    rowkeep = rowsums > threshold
    X = X[rowkeep, :]
    bin_names = [x for (x, v) in zip(bin_names, rowkeep) if v]
    return((X, bin_names))

def filter_slice(intensities, retain = 3):
    """
    Filters a "slice" of a spectra by maintaining the most intense peaks

    Args:
    intensities: Slice of a spectra's intensity array
    retain: Number of the largest intensities to keep from the spectra

    Returns:
    Intensity slice from the spectra with only the largest peaks remaining and 
    the rest zeroed out
    """

    # Sorts the indicies by intensity value, high to low
    # Removes "retain" amount of indices from the front to indicate the largest
    # intensities
    zeroidx = np.flip(np.argsort(intensities))[retain:-1]
    # Zeroes all the indices except the ones that were removed above
    intensities[zeroidx] = 0
    
    return(intensities)

def filter_window(spectra, window_size = 50, retain = 3):
    """ Filters a single spectra by removing smaller intensities in defined m/z windows

    Args:
    spectra: spectra to be filtered
    window_size: approximately how big each window should be - not exact because of 
            spectra having decimal values, so it'll get rounded
    retain: number of intensities to keep for each window 

    Returns:
    A filtered version of the spectra passed in
    """
    mzmax = spectra['m/z array'].max()
    
    # Creates a list with m/z "windows" that are approximately "window_size" large
    windows = np.linspace(0, mzmax, int(np.round(mzmax/window_size)))
    
    for index, mz in enumerate(windows):
        # avoid index out of bounds exceptions
        if index + 1 == len(windows):
            break
        # See what m/z values are contained within this array. Use bitwise & to 
        # create a boolean array of all the indices that have m/z values in the current window
        windowsidx = (spectra['m/z array'] > windows[index]) & (spectra['m/z array'] < windows[index+1])
        # Don't do anything if all there are no charges in the window
        if np.sum(windowsidx) == 0:
            continue
        
        # Filters a single "slice" of the spectra - all the intensities corresponding
        # to the m/z charges in the window will be filtered based on how many
        # are expected to be "retained"
        spectra['intensity array'][windowsidx] = filter_slice(spectra['intensity array'][windowsidx], retain = retain)

    return(spectra)

def bin_sparse_csr(intensity_matrix, file, scan_names, bins, max_parent_mass = 850, window_filter=True, filter_window_size=50, filter_window_retain=3):
    min_bin = min(bins)
    max_bin = max(bins)
    bin_size = (max_bin - min_bin) / len(bins)
    reader = mgf.MGF(file)
    base = os.path.basename(file)
    for spectrum_index, spectrum in enumerate(reader):
        scan_names.append(os.path.splitext(base)[0] + "_" + spectrum['params']['scans'])
        if spectrum['params']['pepmass'][0] > max_parent_mass:
            continue
        if len(spectrum['m/z array']) == 0:
            continue
        if window_filter:
            spectrum = filter_window(spectrum, filter_window_size, filter_window_retain)
        for mz, intensity in zip(spectrum['m/z array'], spectrum['intensity array']):
            if mz > max_bin or mz > spectrum['params']['pepmass'][0]:
                continue
            target_bin = math.floor((mz - min_bin)/bin_size)
            intensity_matrix[target_bin-1, spectrum_index] += intensity
    
    return intensity_matrix

def bin_mgf(mgf_files=None,output_file = None, min_bin = 50, max_bin = 850, bin_size = 0.01, max_parent_mass = 850, verbose = False, remove_zero_sum_rows = True, remove_zero_sum_cols = True, window_filter = True, filter_window_size = 50, filter_window_retain = 3, filter_parent_peak = True):
    """ Bins an mgf file 

    Bins an mgf of ms2 spectra and returns a sparse CSR matrix. Operates on either a single or a list of mgf files.

    Args:
    mgf_files: The path of an mgf file, or a list of multiple mgf files.
    output_file: Name of output file in pickle format.
    min_bin: smallest m/z value to be binned.
    max_bin: largest m/z value to be binned.
    bin_size: m/z range in one bin.
    max_parent_mass: Remove ions larger than this.
    verbose: Print debug info.
    remove_zero_sum_rows: Explicitly remove empty rows (bins).
    remove_zero_sum_cols: Explicitly remove spectra where all values were filtered away (columns)
    filter_parent_peak: Remove all ms2 peaks larger than the parent mass
    
    Returns:
    A sparse CSR matrix X, a list of bin names, and a list of spectra names 
    """
    start = time.time()
    bins = np.arange(min_bin, max_bin, bin_size)

    if type(mgf_files) != list:
        mgf_files = [mgf_files]
    
    n_scans = 0
    for file in mgf_files:
        reader0 = mgf.MGF(file)
        n_scans += len([x for x in reader0])

    X = dok_matrix((len(bins), n_scans), dtype=np.float32)
    scan_names = []
    for file in mgf_files:
        X = bin_sparse_csr(X, file, scan_names, bins, max_parent_mass, window_filter, filter_window_size, filter_window_retain)

    X = X.tocsr()
    X_orig_shape = X.shape

    if remove_zero_sum_rows:
        print(X.shape)
        X, row_names_filter = filter_zero_rows(X)
        bins = [x for (x, v) in zip(bins, row_names_filter) if v]
        print("Removed %s rows" % (X_orig_shape[0] - X.shape[0] )) if verbose else None

    if remove_zero_sum_cols:
        X, col_names_filter = filter_zero_cols(X)
        scan_names = [x for (x, v) in zip(scan_names, col_names_filter) if v]
        print("Removed %s cols" % (X_orig_shape[1] - X.shape[1] )) if verbose else None
        
    if verbose:
            print("Binned in %s seconds with dimensions %sx%s, %s nonzero entries (%s)" % (time.time()-start, X.shape[0], X.shape[1], X.count_nonzero(), X.count_nonzero()/(n_scans*len(bins))))

    if output_file is not None:
        pkl.dump((X, bins, scan_names),open( output_file, "wb"))
    return(X, bins, scan_names)




