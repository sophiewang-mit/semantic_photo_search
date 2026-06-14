import numpy as np

def make_features(text_vec: np.ndarray, image_vec: np.ndarray) -> np.ndarray:
    """Build a feature vector for one (query, candidate) pair.
    Both inputs are L2-normalized CLIP embeddings (512-dimensions)."""
    t = text_vec.astype("float32")
    im = image_vec.astype("float32")
    # CLIP's similarity score
    cosine = np.dot(t, im)
    # interaction per dimension
    elementwise = t* im
    # diff: small diff -> good match
    abs_diff=np.abs(t - im)
    return np.concatenate([[cosine], elementwise, abs_diff]).astype("float32")