import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    preds = np.array(predictions)  # shape: (T, N)
    T, N = preds.shape
    
    result = []
    
    for i in range(N):
        # get all votes for sample i
        votes = preds[:, i]
        
        # count occurrences
        classes, counts = np.unique(votes, return_counts=True)
        
        # find max count
        max_count = np.max(counts)
        
        # filter classes with max count and pick smallest
        winners = classes[counts == max_count]
        result.append(int(np.min(winners)))
    
    return result