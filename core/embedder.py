"""
Embedding interface. SentenceTransformerEmbedder is the real one used in
production/ingestion. HashEmbedder is a deterministic, dependency-free stand-in
used ONLY by this project's own test suite (tests/), because the sandbox this
project was originally scaffolded in has no route to huggingface.co to
download real model weights. Do not use HashEmbedder for anything that needs
actual semantic similarity - it has none.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from core.config import config


class BaseEmbedder(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Returns an (n_texts, dim) float32 array."""
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    """Production embedder. Requires internet access on first run to download
    the model (cached locally afterwards)."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = SentenceTransformer(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


class HashEmbedder(BaseEmbedder):
    """
    Deterministic bag-of-words hashing embedder with NO semantic understanding.
    Same word overlap -> some similarity signal, but nothing like a real model.
    Used only so the retrieval/storage/query PLUMBING can be exercised offline
    (dimension consistency, pgvector storage, cosine ranking mechanics).
    Swap for SentenceTransformerEmbedder before actually evaluating retrieval
    quality - HashEmbedder will make vector search look worse than it is.
    """

    def __init__(self, dim: int = 384, seed: int = 42):
        self.dim = dim
        self.seed = seed

    def _hash_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            h = int(hashlib.md5(f"{self.seed}:{word}".encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._hash_vector(t) for t in texts])


def get_embedder(use_stub: bool = False) -> BaseEmbedder:
    """Factory. use_stub=True is for local/offline testing only."""
    if use_stub:
        return HashEmbedder(dim=config.EMBEDDING_DIM)
    return SentenceTransformerEmbedder()
