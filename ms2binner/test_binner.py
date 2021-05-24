import bin
import scipy.sparse
import numpy as np

"""
Unit tests for test_filter_zero_cols() with filtering 1, 4, and 0 columns, and
then filtering 1 column, but with floats as the values in the matrix
"""
def test_filter_zero_cols_1col():
        M = np.array([[1,0,0,4,5],[4,5,0,0,6],[0,4,0,8,0]])
        V = bin.filter_zero_cols(scipy.sparse.csr_matrix(M))
        assert np.isclose(V[0].toarray(),scipy.sparse.csr_matrix(np.array([[1,0,4,5],[4,5,0,6],[0,4,8,0]])).toarray()).all()
        assert V[1].all() == np.array([True, True, False, True, True]).all()

def test_filter_zero_cols_4cols():
        M = np.array([[0,1,0,0,4,0,5,0],[0,4,5,0,0,0,6,0],[0,0,4,0,8,0,0,0]])
        V = bin.filter_zero_cols(scipy.sparse.csr_matrix(M))
        assert np.isclose(V[0].toarray(), scipy.sparse.csr_matrix(np.array([[1,0,4,5],[4,5,0,6],[0,4,8,0]])).toarray()).all()
        assert V[1].all() == np.array([False, True, True, False, True, False, True, False]).all()

def test_filter_zero_cols_0cols():
        M = np.array([[1,0,1,4,5],[0,0,0,0,0],[0,4,0,8,0]])
        V = bin.filter_zero_cols(scipy.sparse.csr_matrix(M))
        assert np.isclose(V[0].toarray(), scipy.sparse.csr_matrix(np.array([[1,0,1,4,5],[0,0,0,0,0],[0,4,0,8,0]])).toarray()).all()
        assert V[1].all() == np.array([True, True, True, True, True]).all()

def test_filter_zero_cols_1col_float():
        M = np.array([[.001,0,1e-20,.0004,.00005,],[.004,.000005,0,0,.06],[0,.4,0,.0000008,0]])
        V = bin.filter_zero_cols(scipy.sparse.csr_matrix(M))
        assert np.isclose(V[0].toarray(),scipy.sparse.csr_matrix(np.array([[.001,0,.0004,.00005],[.004,.000005,0,.06],[0,.4,.0000008,0]])).toarray()).all()
        assert V[1].all() == np.array([True, True, False, True, True]).all()

"""
Unit tests for test_filter_zero_rows() with filtering 1, 4, and 0 rows, and
then filtering 1 row, but with floats as the values in the matrix
"""
def test_filter_zero_rows_1row():
        M = np.array([[1,0,0,4,5],[0,0,0,0,0],[4,5,0,0,6],[0,4,0,8,0]])
        V = bin.filter_zero_rows(scipy.sparse.csr_matrix(M))
        assert np.isclose(V[0].toarray(),scipy.sparse.csr_matrix(np.array([[1,0,0,4,5],[4,5,0,0,6],[0,4,0,8,0]])).toarray()).all()
        assert V[1].all() == np.array([True, False, True, True]).all()

def test_filter_zero_rows_4rows():
        M = np.array([[0,0,0],[4,0,7],[0,0,5],[0,0,0],[0,0,0],[7,54,2],[0,0,0]])
        V = bin.filter_zero_rows(scipy.sparse.csr_matrix(M))
        assert np.isclose(V[0].toarray(), scipy.sparse.csr_matrix(np.array([[4,0,7],[0,0,5],[7,54,2]])).toarray()).all()
        assert V[1].all() == np.array([False, True, True, False, False, True, False]).all()

def test_filter_zero_rows_0rows():
        M = np.array([[1,0,0,4,5],[4,5,0,0,6],[0,4,0,8,0]])
        V = bin.filter_zero_rows(scipy.sparse.csr_matrix(M))
        assert np.isclose(V[0].toarray(), scipy.sparse.csr_matrix(np.array([[1,0,0,4,5],[4,5,0,0,6],[0,4,0,8,0]])).toarray()).all()
        assert V[1].all() == np.array([True, True, True, True, True]).all()

def test_filter_zero_rows_1row_float():
        M = np.array([[.001,0,0,.0004,.00005,],[.004,.000005,0,0,.06],[1e-20,0,0,0,0],[0,.4,0,.0000008,0]])
        V = bin.filter_zero_rows(scipy.sparse.csr_matrix(M))
        assert np.isclose(V[0].toarray(),scipy.sparse.csr_matrix(np.array([[.001,0,0,.0004,.00005],[.004,.000005,0,0,.06],[0,.4,0,.0000008,0]])).toarray()).all()
        assert V[1].all() == np.array([True, True, False, True, True]).all()

"""
Unit test for filtering rows and keeping bins
"""
def test_row_filter_intensity():
        M = np.array([[.001,0,0,.0004,.00005,],[.004,.000005,0,0,.06],[1e-20,0,0,0,0],[0,.4,0.5,.0000008,0]])
        b = np.array(['bin1', 'bin2', 'bin3', 'bin4', 'bin5'])
        V = bin.row_filter_intensity(M,b)
        assert np.isclose(V[0], np.array([[2.00000000e-01, 0.00000000e+00, 0.00000000e+00, 9.98003992e-01, 8.32639467e-04],[8.00000000e-01, 1.24998438e-05, 0.00000000e+00, 0.00000000e+00,9.99167361e-01],[0.00000000e+00, 9.99987500e-01, 1.00000000e+00, 1.99600798e-03,0.00000000e+00]])).all()
        assert V[1] == ['bin1', 'bin2', 'bin4']

"""
Unit tests for filtering slices, testing keeping the 3 highest values and 0 values
"""
def test_filter_slice_ret3():
        M = np.array([4,5,6,2,5,8,73,1])
        assert np.isclose(bin.filter_slice(M), np.array([0,0,6,0,0,8,73,0])).all()

def test_filter_slice_ret0():
        M = np.array([4,5,6,2,5,8,73,1])
        assert np.isclose(bin.filter_slice(M,0), np.array([0,0,0,0,0,0,0,0])).all()

"""
Unit test for filtering based on windows
"""
def test_window_filtering():
        M = np.array([866.001,543,0,434.0004,124.00005,85.004,54.000005,70,0,234.06,646.23,344.4,0.5,509.0000008,450, 156.52, 187, 267.34, 365.223, 487, 589.53])
        I = np.array([6.1e+01, 9.3e+02, 1.2e+03, 8.1e+01, 1.7e+02, 2.8e+02, 1.4e+02,7.3e+01, 8.0e+01, 5.4e+01, 5.0e+01, 7.9e+02, 1.6e+02, 6.2e+01,6.5e+01, 5.3e+01, 9.0e+01, 5.2e+04, 6.4e+01, 8.3e+03, 3.8e+02])
        S = {'m/z array': M, 'intensity array': I}
        V = bin.filter_window(S, window_size=100, retain=1)
        assert np.isclose(V['m/z array'],np.array([8.66001000e+02,5.43000000e+02, 0.00000000e+00, 4.34000400e+02,1.24000050e+02, 8.50040000e+01, 5.40000050e+01, 7.00000000e+01,0.00000000e+00, 2.34060000e+02, 6.46230000e+02, 3.44400000e+02,5.00000000e-01, 5.09000001e+02, 4.50000000e+02, 1.56520000e+02,1.87000000e+02, 2.67340000e+02, 3.65223000e+02, 4.87000000e+02,5.89530000e+02])).all()
        assert np.isclose(V['intensity array'],np.array([61,930,1200,0,170,280,0,0,80,0,0,790,0,0,0,0,0,52000,0,8300,0])).all()

"""
Test binning a single and multiple mgf files (inherently tests bin_sparse as well)
"""
def test_bin_mgf_single():
        V = bin.bin_mgf("test/spectra.mgf")
        assert np.isclose(V[0].toarray(), np.array([[9.3e+02],[1.2e+03],[2.8e+02],[5.0e+01],[7.9e+02],[5.2e+04]])).all()
        assert np.isclose(V[1], np.array([83.06999999999343, 84.06999999999323, 97.08999999999062, 140.12999999998206, 238.23999999996255, 256.24999999995896])).all()
        assert V[2] == ['spectra_7']

def test_bin_mgf_multiple():
        V = bin.bin_mgf(["test/spectra.mgf", "test/spectra.mgf"])
        assert np.isclose(V[0].toarray(), np.array([[1.86e+03],[2.40e+03],[5.60e+02],[1.00e+02],[1.58e+03],[1.04e+05]])).all()
        assert np.isclose(V[1], np.array([83.06999999999343, 84.06999999999323, 97.08999999999062, 140.12999999998206, 238.23999999996255, 256.24999999995896])).all()
        assert V[2] == ['spectra_7']