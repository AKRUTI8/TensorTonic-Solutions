import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pe = np.zeros((seq_len, d_model), dtype=float)

    # number of sine slots (includes extra one if d_model is odd)
    k = (d_model + 1) // 2
    d_half = d_model // 2  # number of cosine pairs

    # positions (seq_len, 1)
    pos = np.arange(seq_len, dtype=float).reshape(-1, 1)

    # frequency terms
    i = np.arange(k, dtype=float)
    inv_freq = 1.0 / (base ** (2 * i / d_model))  # (k,)

    # angles (seq_len, k)
    angles = pos * inv_freq

    # sine for all even indices
    pe[:, 0:2*k:2] = np.sin(angles)

    # cosine for valid odd indices (only first d_half pairs)
    if d_half > 0:
        pe[:, 1:2*d_half:2] = np.cos(angles[:, :d_half])

    return pe