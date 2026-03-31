import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)
    
    # Handle empty input
    if y.size == 0:
        return 0.0
    
    # Get class counts
    _, counts = np.unique(y, return_counts=True)
    
    # Convert to probabilities
    probs = counts / counts.sum()
    
    # Remove zero probabilities (for stability)
    probs = probs[probs > 0]
    
    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)