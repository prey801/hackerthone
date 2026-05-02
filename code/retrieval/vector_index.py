import faiss
import numpy as np
import os
import pickle

class VectorIndex:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.chunks = []
        
    def build(self, chunks):
        self.chunks = chunks
        if not chunks:
            return
            
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        # IndexFlatIP is inner product; with normalized vectors it's cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
    def save(self, cache_dir):
        with open(os.path.join(cache_dir, "vector_chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        faiss.write_index(self.index, os.path.join(cache_dir, "vector.index"))
        
    def load(self, cache_dir):
        with open(os.path.join(cache_dir, "vector_chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        self.index = faiss.read_index(os.path.join(cache_dir, "vector.index"))
        
    def get_scores(self, query):
        if not self.index:
            return [0.0] * len(self.chunks)
            
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        k = len(self.chunks)
        if k == 0:
            return []
            
        D, I = self.index.search(query_embedding, k)
        
        scores = np.zeros(k)
        for i, idx in enumerate(I[0]):
            scores[idx] = D[0][i]
            
        return scores.tolist()
