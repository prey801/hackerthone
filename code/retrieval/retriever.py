import os
import numpy as np
from .embedder import Embedder
from .bm25_index import BM25Index
from .vector_index import VectorIndex

import pickle

# Layer 3 constants
_RACE_THRESHOLD = 0.10   # minimum normalised score each retriever must clear
_RACE_PENALTY   = 0.20   # score penalty applied when only one retriever fires

# Layer 4 constants
_TIE_EPSILON    = 0.02   # scores within this band are considered tied
_TIE_BOOST      = 0.005  # small bump per matching query token in source name


class HybridRetriever:
    def __init__(self, chunks, cache_dir=".cache"):
        self.chunks = chunks
        self.embedder = Embedder()
        self.bm25 = BM25Index()
        self.vector_index = VectorIndex(self.embedder)
        
        from sentence_transformers import CrossEncoder
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        os.makedirs(cache_dir, exist_ok=True)
        chunks_cache_path = os.path.join(cache_dir, "chunks.pkl")
        
        if os.path.exists(chunks_cache_path):
            with open(chunks_cache_path, "rb") as f:
                cached_chunks = pickle.load(f)
            if len(cached_chunks) == len(chunks):
                print("Loading retrieval indexes from cache...")
                self.bm25.load(cache_dir)
                self.vector_index.load(cache_dir)
                print("Indexes loaded from cache.")
                return
                
        print("Building retrieval indexes...")
        self.bm25.build(chunks)
        self.vector_index.build(chunks)
        
        print("Saving indexes to cache...")
        with open(chunks_cache_path, "wb") as f:
            pickle.dump(chunks, f)
        self.bm25.save(cache_dir)
        self.vector_index.save(cache_dir)
        print("Indexes built and cached successfully.")

    # ------------------------------------------------------------------
    # Layer 3 – retrieval score race
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_score_race(score, bm25_val, dense_val):
        """
        Both retrievers must independently clear _RACE_THRESHOLD.
        If only one fires, apply a proportional penalty.
        """
        bm25_fires  = bm25_val  >= _RACE_THRESHOLD
        dense_fires = dense_val >= _RACE_THRESHOLD
        if bm25_fires and dense_fires:
            return score                        # corroborated – no change
        if not bm25_fires and not dense_fires:
            return score * (1 - _RACE_PENALTY * 2)   # neither fires – heavy penalty
        return score * (1 - _RACE_PENALTY)           # only one fires – light penalty

    # ------------------------------------------------------------------
    # Layer 4 – subject tiebreaker
    # ------------------------------------------------------------------
    @staticmethod
    def _subject_boost(source, query_tokens):
        """
        Count how many unique query tokens appear in the source filename
        (case-insensitive, stem by splitting on non-alpha).  Each match
        contributes _TIE_BOOST to the score.
        """
        filename = os.path.basename(source).lower()
        # simple alphanum tokenisation of the filename
        name_tokens = set(part for part in __import__('re').split(r'[^a-z0-9]', filename) if part)
        matches = sum(1 for t in query_tokens if t in name_tokens)
        return matches * _TIE_BOOST

    def retrieve(self, query, domain_filter=None, top_k=5):
        if not self.chunks:
            return [], 0.0

        # ── Layer 1: BM25 sparse scoring ──────────────────────────────
        bm25_scores = self.bm25.get_scores(query)

        # ── Layer 2: Dense vector scoring ─────────────────────────────
        dense_scores = self.vector_index.get_scores(query)

        # Normalise BM25 to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        norm_bm25 = [s / max_bm25 for s in bm25_scores]

        # Normalise Dense to [0, 1]  (cosine similarity is [-1, 1])
        norm_dense = [(s + 1) / 2 for s in dense_scores]

        # Hybrid fusion: 40% BM25, 60% Dense
        hybrid_scores = []
        for i in range(len(self.chunks)):
            score = (norm_bm25[i] * 0.4) + (norm_dense[i] * 0.6)

            # ── Layer 3: retrieval score race ──────────────────────────
            score = self._apply_score_race(score, norm_bm25[i], norm_dense[i])

            hybrid_scores.append((i, score))

        # Domain filter
        if domain_filter:
            filtered_scores = [
                (i, score) for i, score in hybrid_scores
                if self.chunks[i].get('domain') == domain_filter
            ]
        else:
            filtered_scores = hybrid_scores

        # Sort descending by score
        filtered_scores.sort(key=lambda x: x[1], reverse=True)

        # ── Layer 4: subject tiebreaker ────────────────────────────────
        import re
        query_tokens = set(t for t in re.split(r'[^a-z0-9]', query.lower()) if t)
        if filtered_scores:
            leader_score = filtered_scores[0][1]
            tie_band = [
                (i, score) for i, score in filtered_scores
                if (leader_score - score) <= _TIE_EPSILON
            ]
            rest = filtered_scores[len(tie_band):]

            # Apply subject boost only within the tie band
            tie_band_boosted = [
                (i, score + self._subject_boost(
                    self.chunks[i].get('source', ''), query_tokens))
                for i, score in tie_band
            ]
            tie_band_boosted.sort(key=lambda x: x[1], reverse=True)
            filtered_scores = tie_band_boosted + rest

        # Get more candidates for reranking
        top_k_indices = filtered_scores[:top_k * 2]

        results = []
        for idx, score in top_k_indices:
            chunk_data = self.chunks[idx].copy()
            chunk_data['score'] = score
            results.append(chunk_data)

        # ── Layer 5: Cross-Encoder Reranking ────────────────────────────
        if results:
            pairs = [[query, res['text']] for res in results]
            ce_scores = self.cross_encoder.predict(pairs)
            for res, ce_score in zip(results, ce_scores):
                res['ce_score'] = float(ce_score)
            results.sort(key=lambda x: x['ce_score'], reverse=True)

        results = results[:top_k]
        max_score = results[0].get('ce_score', results[0]['score']) if results else 0.0
        return results, max_score

