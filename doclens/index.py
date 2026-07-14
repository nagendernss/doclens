"""Hand-built in-memory cosine-similarity index (numpy)."""

from __future__ import annotations

import numpy as np

from .types import Chunk, Retrieved


def _normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero vectors stay as zeros (no NaN)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class VectorIndex:
    """In-memory cosine-similarity search via normalized matrix dot product."""

    def __init__(self) -> None:
        """Initialize empty index."""
        self._chunks: list[Chunk] = []
        self._mat: np.ndarray | None = None

    def __len__(self) -> int:
        """Return number of indexed chunks."""
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Add chunks and vectors; L2-normalize; raise ValueError on dim mismatch.

        Args:
            chunks: list of Chunk objects.
            vectors: list of embedding vectors (lists of floats).

        Raises:
            ValueError: if len(chunks) != len(vectors) or embedding dimension mismatch.

        """
        if len(chunks) != len(vectors):
            msg = f"{len(chunks)} chunks but {len(vectors)} vectors"
            raise ValueError(msg)
        if not chunks:
            return
        mat = _normalize(np.asarray(vectors, dtype=np.float64))
        if self._mat is not None and mat.shape[1] != self._mat.shape[1]:
            msg = "embedding dimension mismatch"
            raise ValueError(msg)
        self._mat = mat if self._mat is None else np.vstack([self._mat, mat])
        self._chunks.extend(chunks)

    def search(self, vector: list[float], k: int = 5) -> list[Retrieved]:
        """Search for top-k nearest chunks by cosine similarity.

        Args:
            vector: query embedding (list of floats).
            k: number of results to return (default 5).

        Returns:
            list of Retrieved objects sorted by score descending.

        """
        if self._mat is None or not len(self._chunks):
            return []
        q = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm:
            q = q / norm
        scores = self._mat @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [Retrieved(chunk=self._chunks[i], score=float(scores[i])) for i in order]
