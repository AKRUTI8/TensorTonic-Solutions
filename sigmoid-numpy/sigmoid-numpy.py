import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    Works for scalars, lists, and NumPy arrays.
    """
    x = np.asarray(x, dtype=float)   # convert input to NumPy array
    return 1 / (1 + np.exp(-x))