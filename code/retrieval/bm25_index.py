import os
import pickle
import re
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        
    def build(self, chunks):
        """
        chunks: List of dicts [{"text": "...", "domain": "...", "chunk_id": ...}]
        """
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(chunk['text']) for chunk in chunks]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            
    def save(self, cache_dir):
        with open(os.path.join(cache_dir, "bm25.pkl"), "wb") as f:
            pickle.dump((self.chunks, self.bm25), f)
            
    def load(self, cache_dir):
        with open(os.path.join(cache_dir, "bm25.pkl"), "rb") as f:
            self.chunks, self.bm25 = pickle.load(f)
            
    def _tokenize(self, text):
        # Improved tokenizer: strips punctuation and lowercases
        return re.sub(r'[^\w\s]', '', text).lower().split()
        
    def get_scores(self, query):
        if not self.bm25:
            return [0.0] * len(self.chunks)
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        return scores
