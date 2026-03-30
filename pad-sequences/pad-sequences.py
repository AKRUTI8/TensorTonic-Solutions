import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
    """
    # Handle empty input
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=int)
    
    # Determine max_len
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)
    
    N = len(seqs)
    L = max_len
    
    # Initialize result array with pad_value
    result = np.full((N, L), pad_value, dtype=int)
    
    # Fill with sequence values (truncate if needed)
    for i, seq in enumerate(seqs):
        length = min(len(seq), L)
        if length > 0:
            result[i, :length] = seq[:length]
    
    return result