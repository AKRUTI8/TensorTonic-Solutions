import numpy as np

def matrix_transpose(A):
    # Convert list to NumPy array
    A = np.array(A)
    
    N, M = A.shape
    
    transpose = np.zeros((M, N), dtype=A.dtype)
    
    for i in range(N):
        for j in range(M):
            transpose[j][i] = A[i][j]
    
    return transpose