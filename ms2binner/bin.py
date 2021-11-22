"""
Authors: Asker Brejnrod & Arjun Sampath

This file contains useful methods for binning ms2 spectra along with filtering out noise from
the resultant intensity matrix

Notable libraries used are:
    - pyteomics: https://pyteomics.readthedocs.io/en/latest/
    - numpy: https://numpy.org/doc/stable/
    - scipy: https://docs.scipy.org/doc/scipy/reference/
    - tqdm: https://tqdm.github.io
    - sqlite3: https://docs.python.org/3/library/sqlite3.html
"""

from pyteomics import mgf
import numpy as np
from scipy.sparse import dok_matrix
import math
import time
import pickle as pkl
import os
import glob
import tqdm

def filter_zero_cols(csr, threshold=1e-20):
    """ Removes all columns that only contain zeroes

    Args:
        csr: Input CSR matrix to filter
        threshold: Threshold for summed cols to be bigger than, default is 1e-20

    Returns:
        A sparse CSR matrix with zero-sum columns filtered out and a boolean array 
            indicating whether to "keep" each column
    
    """
    # Sums each column and creates a boolean array of whether each 
    # summed column is greater than 0
    keep = np.array(csr.sum(axis = 0) > threshold).flatten()
    # Only keeps columns that have a corresponding True value in keep
    csr = csr[:,keep]

    return(csr, keep)

def filter_zero_rows(csr, threshold=1e-20):
    """ Removes all rows that only contain zeroes

    Args:
        csr: Input CSR matrix to filter
        threshold: Threshold for summed rows to be bigger than, default is 1e-20

    Returns:
        A sparse CSR matrix that has all zero-sum rows filtered out and a boolean
            array indicating whether to "keep" each row
    
    """

    # Sums each row and creates a boolean array of whether each 
    # summed row is greater than 0
    keep = np.array(csr.sum(axis = 1) > threshold).flatten()
    # Only keeps rows that have a corresponding True value in keep
    csr = csr[keep]

    return(csr, keep)

def row_filter_intensity(X, bin_names, threshold = 1/100):
    """ Filters the rows of an intensity (bins x spectra) matrix based on the given threshold
    
    Args:
        X: Numpy or Scipy matrix with bins as the rows and spectra as the columns. Columns are expected to not sum to zero
        bin_names: Array of bins that corresponds to the rows of X
        threshold: Value to filter based off

    Returns:
        Returns a tuple with the filtered X matrix and the array of filtered bins
    
    """

    # Sums up all the columns into an array
    colsums = np.array(X.sum(axis = 0)).flatten()

    # Loops through all the columns and normalizes them based on the column's sum
    for i in range(X.shape[1]):
        X[:, i] = X[:, i]/colsums[i]
    
    # Sums up all the rows into an array
    rowsums = np.array(X.sum(axis = 1)).flatten()
    # Determine which rows to keep based on the sums of the normalized values relative
    # to the indicated threshold
    rowkeep = rowsums > threshold
    X = X[rowkeep, :]
    # Only keep the specified bins as well to match the filtered rows
    bin_names = [x for (x, v) in zip(bin_names, rowkeep) if v]

    return((X, bin_names))

def filter_slice(intensities, retain = 3):
    """ Filters a "slice" of a spectra by maintaining the most intense peaks

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
    zeroidx = np.flip(np.argsort(intensities))[retain:]
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

def bin_sql(df, output_file = None, min_bin = 50, max_bin = 850, bin_size = 0.01, max_parent_mass = 850, verbose = False, remove_zero_sum_rows = True, remove_zero_sum_cols = True, window_filter = True, filter_window_size = 50, filter_window_retain = 3, filter_parent_peak = True):
    start = time.time()
    bins = np.arange(min_bin, max_bin, bin_size)

    scan_names = []
    # Create an empty sparse matrix with bins as the rows and spectra as the columns
    X = dok_matrix((len(bins), n_scans), dtype=np.float32)

    spectra_list = df.to_numpy()
    base = os.path.basename("sql_db")

    for spectra in spectra_list:
        idx = spectra[0]
        pmass = spectra[1]
        mz = spectra[2]
        intens = spectra[3]
        scan_names.append(os.path.splitext(base)[0] + "_" + str(idx))
        X =  bin_arrays(X, intens, mz, pmass, idx, bins, verbose=verbose)

    # Convert from DOK to CSR for easier processing/handling
    X = X.tocsr()
    X_orig_shape = X.shape

    # Filter out rows summing to zero if specified
    print("\nSummary:") if verbose else None
    if remove_zero_sum_rows:
        X, row_names_filter = filter_zero_rows(X)
        # Adjust the bins accordingly based on row_names_filter which says which rows to keep
        bins = [x for (x, v) in zip(bins, row_names_filter) if v]
        print("Removed %s rows" % (X_orig_shape[0] - X.shape[0] )) if verbose else None

    # Filter out columns summing to zero if specified
    if remove_zero_sum_cols:
        X, col_names_filter = filter_zero_cols(X)
        # Adjust the scan names accordingly based on col_names_filter which says which columns to keep
        scan_names = [x for (x, v) in zip(scan_names, col_names_filter) if v]
        print("Removed %s cols" % (X_orig_shape[1] - X.shape[1] )) if verbose else None
        
    if verbose:
            print("Binned in %s seconds with dimensions %sx%s, %s nonzero entries (%s)\n" % (time.time()-start, X.shape[0], X.shape[1], X.count_nonzero(), X.count_nonzero()/(n_scans*len(bins))))

    # If an output file is specified, write to it
    if output_file is not None:
        # Use pickle to create a binary file that holds the intensity matrix, bins, and spectra names
        pkl.dump((X, bins, scan_names),open( output_file, "wb"))
        print("Wrote data to " + output_file) if verbose else None
    return(X, bins, scan_names)

def bin_arrays(X,intensity_array,mz_array, parent_mass, spectrum_index,bins, max_parent_mass = 850, verbose = False):
    # Do a basic filter based on the mass of the spectra
    if parent_mass > max_parent_mass:
        return
    # Get min and max bins
    min_bin = min(bins)
    max_bin = max(bins)
    # Determine bin size from bins array
    bin_size = (max_bin - min_bin) / len(bins)
    # Loop through all the charges in the spectra and get the corresponding intensities
    for mz, intensity in zip(spectrum['m/z array'], spectrum['intensity array']):
        # If the charge is outside of the max bin specified or if it's too large 
        # relative to the spectra itself, skip it
        if mz > max_bin or mz > spectrum['params']['pepmass'][0]:
            continue
        # Figure out what the index of the bin should be
        target_bin = math.floor((mz - min_bin)/bin_size)
        # Add the intensity to the right spot in the matrix. Uses target_bin-1 because
        # indices start at 0. Does += (adds) so that if any charges from the same spectra
        # fall into the same bin because of larger bin sizes, it "stacks" on top of each other
        X[target_bin-1, spectrum_index] += intensity

    return X

def bin_sparse(X, file, scan_names, bins, max_parent_mass = 850, verbose=False, window_filter=True, filter_window_size=50, filter_window_retain=3):
    """ Parses and bins a single MGF file into a matrix that holds the charge intensities of all the spectra

    Args:
        X: Scipy sparse matrix in the format of bins on the rows and spectra on the columns
        file: MGF file to read spectra in from
        scan_names: List of spectra names to append to
        bins: Numpy array of a list of bins for holding spectra charges
        max_parent_mass: Threshold value for max mass of the spectra for filtering
        window_filter: Boolean for whether to use a window filter to remove small intensity peaks
        filter_window_size: Size of each window for the window filter
        filter_window_retain: Number of peaks to keep for the window filter

    Returns:
        A tuple containing a Scipy sparse matrix that has all the charges' intensities binned and related
            to their corresponding spectra, along with a list of scan names that indicate the mgf file and spectra
            number corresponding to each column of the intensity matrix
    
    """

    # Get min and max bins
    min_bin = min(bins)
    max_bin = max(bins)
    # Determine bin size from bins array
    bin_size = (max_bin - min_bin) / len(bins)
    # Parse MGF file
    reader = mgf.MGF(file)
    # File's name without extension is used for creating scan names
    base = os.path.basename(file)

    print("Binning " + file) if verbose else None

    length = X.shape[1]
    if verbose:
        pbar = tqdm.tqdm(total=length, unit='spectra', smoothing=0.1, dynamic_ncols=True)

    half = length/2
    curr = 1
    # Go through all the spectra from the MGF file
    for spectrum_index, spectrum in enumerate(reader):
        if verbose:
            pbar.update()
	
        if curr <= half:
                # Create the scan name based on the MGF file and the current spectra number
                scan_names.append(os.path.splitext(base)[0] + "_filtered" + "_" + spectrum['params']['scans'])
        else:
                scan_names.append(os.path.splitext(base)[0] + "_" + spectrum['params']['scans'])
        curr +=1
        # Do a basic filter based on the mass of the spectra
        if spectrum['params']['pepmass'][0] > max_parent_mass:
            continue
        # Some spectra might not have any charges so skip those
        if len(spectrum['m/z array']) == 0:
            continue
        # First do the window filter to remove any noise from low intensity peaks if specified
        if window_filter:
            spectrum = filter_window(spectrum, filter_window_size, filter_window_retain)

        # Loop through all the charges in the spectra and get the corresponding intensities
        for mz, intensity in zip(spectrum['m/z array'], spectrum['intensity array']):
            # If the charge is outside of the max bin specified or if it's too large 
            # relative to the spectra itself, skip it
            if mz > max_bin or mz > spectrum['params']['pepmass'][0]:
                continue
            # Figure out what the index of the bin should be
            target_bin = math.floor((mz - min_bin)/bin_size)
            # Add the intensity to the right spot in the matrix. Uses target_bin-1 because
            # indices start at 0. Does += (adds) so that if any charges from the same spectra
            # fall into the same bin because of larger bin sizes, it "stacks" on top of each other
            X[target_bin-1, spectrum_index] += intensity

    if verbose:
        pbar.close()

    # Normalize the matrix, making each bin relative to its max value
    for idx in range(0, X.shape[0]):
        X[idx] = X[idx]/X[idx].toarray().max()

    print("Finished Binning " + file) if verbose else None
    
    return (X,scan_names)

def bin_mgf(mgf_files,output_file = None, min_bin = 50, max_bin = 850, bin_size = 0.01, max_parent_mass = 850, verbose = False, remove_zero_sum_rows = True, remove_zero_sum_cols = True, window_filter = True, filter_window_size = 50, filter_window_retain = 3, filter_parent_peak = True):
    """ Bins an mgf file 

    Bins an mgf of ms2 spectra and returns a sparse CSR matrix. Operates on either a single or a list of mgf files.
    The CSR matrix has bins on the rows and spectra as the columns

    Args:
        mgf_files: The path of an mgf file, or a list of multiple mgf files. Can be a directory path containing mgf files
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
    # Creates a list of bins based on the parameters inputted
    bins = np.arange(min_bin, max_bin, bin_size)
 
    # If the path passed in is a directory then loop through it
    if type(mgf_files) != list and os.path.isdir(mgf_files):
        dir = mgf_files
        mgf_files = []
        directory = os.fsencode(dir)
        
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            # only save filenames of .mgf files in the directory
            if filename.endswith(".mgf"): 
                mgf_files.append(os.path.join(dir, filename))

    # If only one mgf file is passed in, make it a list so that it's iterable
    elif type(mgf_files) != list:
        mgf_files = glob.glob(mgf_files)
    
    n_scans = 0
    # Go through all the mgf files and see how many spectra are there in total
    # for construction of the intensity matrix X
    for file in mgf_files:
        reader0 = mgf.MGF(file)
        n_scans += len([x for x in reader0])

    scan_names = []
    # Create an empty sparse matrix with bins as the rows and spectra as the columns
    X = dok_matrix((len(bins), n_scans), dtype=np.float32)
    # Go through each file and bin each MGF file
    for file in mgf_files:
        X,scan_names = bin_sparse(X, file, scan_names, bins, max_parent_mass, verbose, window_filter, filter_window_size, filter_window_retain)

    # Convert from DOK to CSR for easier processing/handling
    X = X.tocsr()
    X_orig_shape = X.shape

    # Filter out rows summing to zero if specified
    print("\nSummary:") if verbose else None
    if remove_zero_sum_rows:
        X, row_names_filter = filter_zero_rows(X)
        # Adjust the bins accordingly based on row_names_filter which says which rows to keep
        bins = [x for (x, v) in zip(bins, row_names_filter) if v]
        print("Removed %s rows" % (X_orig_shape[0] - X.shape[0] )) if verbose else None

    # Filter out columns summing to zero if specified
    if remove_zero_sum_cols:
        X, col_names_filter = filter_zero_cols(X)
        # Adjust the scan names accordingly based on col_names_filter which says which columns to keep
        scan_names = [x for (x, v) in zip(scan_names, col_names_filter) if v]
        print("Removed %s cols" % (X_orig_shape[1] - X.shape[1] )) if verbose else None
        
    if verbose:
            print("Binned in %s seconds with dimensions %sx%s, %s nonzero entries (%s)\n" % (time.time()-start, X.shape[0], X.shape[1], X.count_nonzero(), X.count_nonzero()/(n_scans*len(bins))))

    # If an output file is specified, write to it
    if output_file is not None:
        # Use pickle to create a binary file that holds the intensity matrix, bins, and spectra names
        pkl.dump((X, bins, scan_names),open( output_file, "wb"))
        print("Wrote data to " + output_file) if verbose else None
    return(X, bins, scan_names)




