"""
Embedding-backed semantic similarity.

Upgrades the auditor modules from raw TF-IDF word-overlap to real sentence
embeddings (all-MiniLM-L6-v2), which understand meaning rather than just
shared vocabulary. This makes the Hallucination Detector and Bias Checker
meaningfully more accurate.

If sentence-transformers or its model weights can't be loaded (e.g. no
internet on first run), this fails safe: `available` becomes False and
callers fall back to TF-IDF similarity instead of crashing.
"""

import functools
from typing import Optional

_MODEL_NAME = "all-MiniLM-L6-v2"


@functools.lru_cache(maxsize=1)
def _load_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(_MODEL_NAME)
    except Exception:
        return None


class EmbeddingModel:
    """Thin wrapper around a sentence-transformers model with a safe fallback."""

    def __init__(self):
        self._model = _load_model()
        self.available = self._model is not None

    def embed(self, text: str):
        if not self.available:
            return None
        return self._model.encode(text, normalize_embeddings=True)

    def similarity(self, text_a: str, text_b: str) -> Optional[float]:
        """Cosine similarity between two texts' embeddings, or None if unavailable."""
        if not self.available or not text_a.strip() or not text_b.strip():
            return None
        import numpy as np
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)
        return float(np.clip(np.dot(vec_a, vec_b), -1.0, 1.0))
