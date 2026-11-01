"""Retrieval helpers for RepoPilot AI.

This package turns text into embeddings and stores those vectors in
FAISS so a question can find nearby code chunks.
"""

from repopilot_ai.retrieval.embedding_generator import (
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_MODEL_NAME,
    EmbeddingGenerator,
)
from repopilot_ai.retrieval.rag import RAGPipeline
from repopilot_ai.retrieval.retriever import CodeRetriever
from repopilot_ai.retrieval.vector_store import VectorStore

__all__ = [
    "CodeRetriever",
    "DEFAULT_DEVICE",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_MODEL_NAME",
    "EmbeddingGenerator",
    "RAGPipeline",
    "VectorStore",
]
