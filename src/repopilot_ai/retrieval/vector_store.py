"""Store code-chunk embeddings in FAISS and search by similarity.

FAISS (Facebook AI Similarity Search) keeps a list of vectors and can
find the nearest ones to a query vector. It does **not** keep our
Python dicts, so this module stores chunk metadata (file path, source
code, line numbers, ...) in a parallel Python list.

We use ``IndexFlatIP``. "Flat" means every stored vector is compared
(no approximation). "IP" means inner product. Our embedding generator
L2-normalizes vectors, so inner product equals cosine similarity:
1.0 is the same direction, 0.0 is unrelated, -1.0 is opposite.

This module never imports or executes repository source. It only
stores dictionaries and float arrays.
"""

from __future__ import annotations

from typing import Any, Sequence

import faiss
import numpy as np

from repopilot_ai.retrieval.embedding_generator import DEFAULT_EMBEDDING_DIMENSION


class VectorStore:
    """In-memory FAISS index plus the chunk metadata that goes with it."""

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")

        self.dimension = dimension
        # Compare every vector with inner product (cosine, if normalized).
        self._index = faiss.IndexFlatIP(dimension)
        # index.ntotal is the count of vectors. Metadata[i] belongs to vector i.
        self._metadata: list[dict[str, Any]] = []

    def __len__(self) -> int:
        """Number of chunks currently stored."""
        return len(self._metadata)

    @property
    def is_empty(self) -> bool:
        return len(self._metadata) == 0

    def add(self, chunks: Sequence[dict[str, Any]]) -> int:
        """Add chunks that already have an ``embedding`` field.

        Returns how many chunks were added. An empty list adds nothing.
        A missing or wrong-shaped embedding raises; we never skip a
        chunk silently.
        """
        records, matrix = _prepare_chunks(chunks, self.dimension)
        if not records:
            return 0

        self._index.add(matrix)
        self._metadata.extend(records)
        return len(records)

    def search(
        self,
        query_embedding: np.ndarray | Sequence[float],
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the ``k`` chunks closest to ``query_embedding``.

        Each hit is ``{"score": float, "chunk": {...}}``. ``score`` is
        the inner product (cosine similarity for normalized vectors).
        Results are ordered most similar first.

        An empty index, or ``k`` larger than the store, is handled
        without FAISS placeholder rows (those would be index ``-1``).
        """
        query = _prepare_query(query_embedding, self.dimension)
        top_k = _require_positive_k(k)
        if self.is_empty:
            return []

        request_k = min(top_k, len(self._metadata))
        scores, indices = self._index.search(query, request_k)

        hits: list[dict[str, Any]] = []
        for score, position in zip(scores[0], indices[0], strict=True):
            if position < 0:
                continue
            hits.append(
                {
                    "score": float(score),
                    "chunk": dict(self._metadata[int(position)]),
                }
            )
        return hits


def _prepare_chunks(
    chunks: Sequence[dict[str, Any]],
    dimension: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Validate chunks and build a float32 matrix for FAISS."""
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be a sequence of dictionaries")
    if not chunks:
        return [], np.empty((0, dimension), dtype=np.float32)

    stored: list[dict[str, Any]] = []
    rows: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise TypeError(
                f"chunks[{index}] must be a dictionary, got {type(chunk).__name__}"
            )
        if "embedding" not in chunk:
            raise ValueError(f"chunks[{index}] is missing required key 'embedding'")
        vector = _as_vector(chunk["embedding"], dimension, label=f"chunks[{index}]['embedding']")
        stored.append(_metadata_without_embedding(chunk))
        rows.append(vector)

    matrix = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)
    return stored, matrix


def _metadata_without_embedding(chunk: dict[str, Any]) -> dict[str, Any]:
    """Copy the chunk, keep source and metadata, drop the float array.

    The vector already lives in FAISS. Copying the dict means later
    edits by the caller cannot change what we stored.
    """
    copied = dict(chunk)
    copied.pop("embedding", None)
    return copied


def _prepare_query(
    query_embedding: np.ndarray | Sequence[float],
    dimension: int,
) -> np.ndarray:
    """Turn a 1-D query into shape ``(1, dimension)`` for FAISS.search."""
    vector = _as_vector(query_embedding, dimension, label="query_embedding")
    return np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)


def _as_vector(value: object, dimension: int, *, label: str) -> np.ndarray:
    """Require a 1-D numeric vector of the index dimension."""
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a numeric array") from exc

    if array.ndim == 2 and array.shape[0] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{label} must be 1-D, got shape {array.shape}")
    if array.shape[0] != dimension:
        raise ValueError(
            f"{label} has dimension {array.shape[0]}, expected {dimension}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return array


def _require_positive_k(k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    return k
