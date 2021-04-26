import bin
import scipy.sparse
import numpy as np

def test_filter_zero_cols():
        D = np.array([[1,0,0,4,5],[4,5,0,0,6],[0,4,0,8,0]])
        assert bin.filter_zero_cols(scipy.sparse.csr_matrix(D)) == scipy.sparse.csr_matrix(np.array([[1,0,4,5],[4,5,0,6],[0,4,8,0]]))