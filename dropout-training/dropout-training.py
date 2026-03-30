import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """

    x = np.array(x, dtype=float)

    if rng is None:
        rng = np.random

    # random mask: 1 = keep, 0 = drop
    mask = (rng.random(x.shape) >= p).astype(float)

    # scale kept units
    dropout_pattern = mask / (1.0 - p)

    output = x * dropout_pattern

    return output, dropout_pattern