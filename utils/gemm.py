import numpy as np
from scipy import sparse

def dense_matrix_multiplication(M,N,K,*args):
    A = np.random.rand(M,N)
    B = np.random.rand(N,K)
    result = A @ B
    return result

def sparse_matrix_multiplication (M,N,K,density,*args):
    A = sparse.random(M, N, density = density, format = 'csr')
    B = np.random.rand(N, K)
    result = A @ B
    return result