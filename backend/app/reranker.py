import pickle, numpy as np
from ml.features import make_fetaures

class Reranker:
    def __init__(self, path = "models/reranker.pkl"):
        with open(path, 'rb') as f:
            self.clf = pickle.load(f)
        
    def rerank(self, text_vec, candidates):
        """Candidates: list of """