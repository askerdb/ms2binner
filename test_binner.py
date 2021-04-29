import bin
import scipy.sparse
import numpy as np

"""
4 unit tests for test_filter_zero_cols() with filtering 1, 4, and 0 columns, and
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
4 unit tests for test_filter_zero_rows() with filtering 1, 4, and 0 rows, and
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

