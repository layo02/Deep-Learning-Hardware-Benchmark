import numpy as np
import cv2
from scipy import sparse

def dense_matrix_multiplication(M,N,K):
    A = np.random.rand(M, N)
    B = np.random.rand(N,K)
    result = A @ B
    return result

def sparse_matrix_multiplication (M,N,K, density):
    A = sparse.random(M, N, density = density, format = 'csr')
    B = np.random.rand(N, K)
    result = A @ B
    return result