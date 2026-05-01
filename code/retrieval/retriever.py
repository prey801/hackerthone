import numpy as np
from .embedder import Embedder
from .bm25_index import BM25Index
from .vector_index import VectorIndex

class HybridRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.embedder = Embedder()
        self.bm25 = BM25Index()
        self.vector_index = VectorIndex(self.embedder)
        
        print("Building retrieval indexes...")
        self.bm25.build(chunks)
        self.vector_index.build(chunks)
        print("Indexes built successfully.")
        
    def retrieve(self, query, domain_filter=None, top_k=5):
        if not self.chunks:
            return [], 0.0
            
        # Get raw scores
        bm25_scores = self.bm25.get_scores(query)
        dense_scores = self.vector_index.get_scores(query)
        
        # Normalize BM25 scores (0 to 1)
        # BM25 scores can be arbitrary positive floats
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        norm_bm25 = [s / max_bm25 for s in bm25_scores]
        
        # Normalize Dense scores (0 to 1)
        # Cosine similarity is [-1, 1]. Shift to [0, 1]
        norm_dense = [(s + 1) / 2 for s in dense_scores]
        
        # Calculate hybrid scores
        hybrid_scores = []
        for i in range(len(self.chunks)):
            # 40% BM25, 60% Dense
            score = (norm_bm25[i] * 0.4) + (norm_dense[i] * 0.6)
            hybrid_scores.append((i, score))
            
        # Filter by domain if specified
        if domain_filter:
            filtered_scores = [
                (i, score) for i, score in hybrid_scores 
                if self.chunks[i].get('domain') == domain_filter
            ]
        else:
            filtered_scores = hybrid_scores
            
        # Sort descending by score
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-K
        top_k_indices = filtered_scores[:top_k]
        
        results = []
        for idx, score in top_k_indices:
            chunk_data = self.chunks[idx].copy()
            chunk_data['score'] = score
            results.append(chunk_data)
            
        max_score = results[0]['score'] if results else 0.0
        return results, max_score
