import pickle, numpy as np
from ml.features import make_features

class Reranker:
    def __init__(self, path = "models/reranker.pkl"):
        with open(path, 'rb') as f:
            self.clf = pickle.load(f)
        
    def rerank(self, text_vec, candidates):
        """Candidates: list of (path, image_vec). 
         Returns reordered list of (path, score). """
        t = text_vec / np.linalg.norm(text_vec)
        feats = np.stack([make_features(t, iv) for _, iv in candidates])
        scores = self.clf.predict_proba(feats)[:, 1]
        order = np.argsort(-scores)
        return [(candidates[i][0], float(scores[i])) for i in order]