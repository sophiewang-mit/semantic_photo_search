import faiss
import numpy as np

class PhotoIndex:
    def __init__(self, dim: int = 512):
        # Inner-product index; normalize vectors for cosine similarity
        # stores vectors
        self.index = faiss.IndexFlatIP(dim)
        # vectors -> filepaths
        self.paths = []
        # normalized image vectors 
        self.vectors = []

    def add(self, vector: np.ndarray, path: str):
        v = vector.astype("float32").reshape(1,-1)
        faiss.normalize_L2(v)
        self.index.add(v)
        self.paths.append(path)
        self.vectors.append(v[0].copy())
    
    def search(self, query_vector: np.ndarray, k: int = 5):
        q = query_vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)
        # find similarity scores
        scores, idx = self.index.search(q, min(k, len(self.paths)))
        return [(self.paths[i],float(scores[0][j])) for j, i in enumerate(idx[0]) if i != -1]

    def search_with_vectors(self, query_vector: np.ndarray, k: int = 100):
        q = query_vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)

        scores, idx = self.index.search(q, min(k, len(self.paths)))

        return [(self.paths[i], float(scores[0][j]), self.vectors[i]) for j, i in enumerate(idx[0]) if i != -1]