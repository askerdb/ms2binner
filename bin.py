from pyteomics import mgf, mzxml
import numpy as np
from scipy.sparse import dok_matrix
import math
import time
import pickle as pkl
import os

def filter_zero_cols(csr):
    keep = np.array(csr.sum(axis = 0) > 0).flatten()
    csr = csr[:,keep]
    return(csr, keep)

def filter_zero_rows(csr):
    keep = np.array(csr.sum(axis = 1) > 0).flatten()
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

def bin_sparse_dok(mgf_file=None, mgf_files=None, output_file = None, min_bin = 50, max_bin = 850, bin_size = 0.01, max_parent_mass = 850, verbose = False, remove_zero_sum_rows = True, remove_zero_sum_cols = True):
    """ Bins an mgf file 

    Bins an mgf of ms2 spectra and returns a sparse dok matrix. Operates on either a single or a list of mgf files.

    Args:
    mgf_file: The path of an mgf file.
    mgf_files: A list of mgf files.
    output_file = Name of output file in pickle format.
    min_bin = smallest m/z value to be binned.
    max_bin = largest m/z value to be binned.
    bin_size: M/z range in one bin.
    max_parent_mass: Remove ions larger than this.
    verbose: Print debug info.
    remove_zero_sum_rows: Explicitly remove empty rows (bins).
    remove_zero_sum_cols: Explicitly remove spectra were all values were filtered away (columns)

    returns:
    A sparse dok matrix X, a list of bin names, and a list of spectra names 
    """
    start = time.time()
    bins = np.arange(min_bin, max_bin, bin_size)

    if mgf_file != None:
        mgf_files = [mgf_file]
    
    n_scans = 0
    for file in mgf_files:
        reader0 = mgf.MGF(file)
        n_scans += len([x for x in reader0])

    X = dok_matrix((len(bins), n_scans), dtype=np.float32)
    scan_names = []
    for file in mgf_files:
        reader = mgf.MGF(file)
        base = os.path.basename(file)
        for spectrum_index, spectrum in enumerate(reader):
            scan_names.append(os.path.splitext(base)[0] + "_" + spectrum['params']['scans'])
            if spectrum['params']['pepmass'][0] > max_parent_mass:
                continue
            if len(spectrum['m/z array']) == 0:
                continue

            for mz, intensity in zip(spectrum['m/z array'], spectrum['intensity array']):
                target_bin = math.floor((mz - min_bin)/bin_size)
                X[target_bin, spectrum_index] += intensity

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

    


