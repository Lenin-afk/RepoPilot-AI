"""Create embeddings for code chunks and natural-language queries.

An embedding model maps text to a list of numbers (a vector). Texts
that mean similar things land close together in that space, so later
search can compare a question to stored chunks with math instead of
keywords.

This module loads Sentence Transformers once per generator instance.
It does not talk to a vector database.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
DEFAULT_EMBEDDING_DIMENSION = 384


class EmbeddingGenerator:
    """Encode text with a Sentence Transformer model.

    The heavy model is loaded in ``__init__`` and reused for every
    later call. Default device is CPU so this works on machines
    without a GPU; pass ``device="cuda"`` (or another Sentence
    Transformers device string) when you want a different target.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")

        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def model(self) -> SentenceTransformer:
        """The loaded Sentence Transformer. Loaded once per instance."""
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Number of numbers in each embedding this model produces."""
        getter = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        dimension = getter()
        if dimension is None:
            raise RuntimeError("The loaded model did not report an embedding dimension")
        return int(dimension)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a list of strings.

        Returns a 2-D NumPy array of shape ``(len(texts), dimension)``.
        An empty list returns shape ``(0, dimension)`` — nothing is
        dropped because there was nothing to encode.
        """
        validated = _require_text_list(texts)
        if not validated:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)
        return self._encode(validated)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one natural-language search question.

        Returns a 1-D NumPy array of shape ``(dimension,)``.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        return self._encode([query])[0]

    def embed_chunks(self, chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Embed code chunks and keep every original field.

        Each output dict is a shallow copy of the input chunk plus an
        ``embedding`` key (a 1-D NumPy array). Chunks are never skipped:
        empty ``content`` is still encoded, and a missing or invalid
        ``content`` field raises instead of being dropped.
        """
        records = _require_chunk_list(chunks)
        if not records:
            return []

        texts = [str(chunk["content"]) for chunk in records]
        vectors = self._encode(texts)

        embedded: list[dict[str, Any]] = []
        for chunk, vector in zip(records, vectors, strict=True):
            item = dict(chunk)
            item["embedding"] = vector
            embedded.append(item)
        return embedded

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Run the model. Vectors are L2-normalized for cosine search."""
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array


def _require_text_list(texts: Sequence[str]) -> list[str]:
    """Reject non-lists and non-string items. Empty lists are allowed."""
    if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings")
    validated: list[str] = []
    for index, item in enumerate(texts):
        if not isinstance(item, str):
            raise TypeError(f"texts[{index}] must be a string, got {type(item).__name__}")
        validated.append(item)
    return validated


def _require_chunk_list(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject bad chunk payloads instead of skipping them."""
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be a sequence of dictionaries")
    validated: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise TypeError(
                f"chunks[{index}] must be a dictionary, got {type(chunk).__name__}"
            )
        if "content" not in chunk:
            raise ValueError(f"chunks[{index}] is missing required key 'content'")
        if not isinstance(chunk["content"], str):
            raise TypeError(
                f"chunks[{index}]['content'] must be a string, "
                f"got {type(chunk['content']).__name__}"
            )
        validated.append(chunk)
    return validated
