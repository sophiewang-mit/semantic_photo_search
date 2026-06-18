import numpy as np

def margin(sorted_scores):
    """Gap between top 1 and 2 CLIP cosine scores.
    Low = uncertain"""
    if len(sorted_scores) < 2:
        return 1.0
    return float(sorted_scores[0] - sorted_scores[1])

def entropy(sorted_scores, k = 10, temp = 0.1):
    """Softmax entropy over top-k scores. 
    High = uncertain"""
    s = np.asarray(sorted_scores[:k], dtype = "float64") / temp
    s -= s.max()
    p = np.exp(s)
    p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())