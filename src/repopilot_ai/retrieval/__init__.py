"""Retrieval helpers for RepoPilot AI.

This package turns text (code chunks and search questions) into
numerical embeddings. Vector storage comes later; this package only
creates the vectors.
"""

from repopilot_ai.retrieval.embedding_generator import (
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_MODEL_NAME,
    EmbeddingGenerator,
)

__all__ = [
    "DEFAULT_DEVICE",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_MODEL_NAME",
    "EmbeddingGenerator",
]
