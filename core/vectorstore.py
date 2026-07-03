"""
Lightweight vector store built on TF-IDF + cosine similarity.
"""

import os
import re
import glob
from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


def _split_into_chunks(text: str, max_words: int = 80) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for p in paragraphs:
        words = p.split()
        if len(words) <= max_words:
            chunks.append(p)
        else:
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
    return chunks


class VectorStore:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None

    def load_directory(self, directory: str):
        paths = sorted(glob.glob(os.path.join(directory, "*.txt")))
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            source_name = os.path.basename(path)
            for i, chunk_text in enumerate(_split_into_chunks(text)):
                self.chunks.append(Chunk(text=chunk_text, source=source_name, chunk_id=i))
        self._build_index()

    def _build_index(self):
        if not self.chunks:
            raise ValueError("No documents loaded into vector store.")
        texts = [c.text for c in self.chunks]
        self._matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 3) -> List[RetrievalResult]:
        if self._matrix is None:
            raise ValueError("Vector store index not built. Call load_directory() first.")
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [RetrievalResult(chunk=self.chunks[i], score=float(sims[i])) for i in top_idx]

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Semantic similarity between two texts. Prefers real sentence
        embeddings (understands meaning, not just shared words); falls back
        to TF-IDF cosine similarity against this store's fitted vocabulary
        if the embedding model isn't available.
        """
        from core.embeddings import EmbeddingModel

        if not hasattr(self, "_embedder"):
            self._embedder = EmbeddingModel()

        emb_score = self._embedder.similarity(text_a, text_b)
        if emb_score is not None:
            return emb_score

        try:
            vecs = self.vectorizer.transform([text_a, text_b])
            return float(cosine_similarity(vecs[0], vecs[1])[0][0])
        except Exception:
            return 0.0
