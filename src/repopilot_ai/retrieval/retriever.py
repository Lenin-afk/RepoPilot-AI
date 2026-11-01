"""End-to-end retrieval: repository on disk -> ranked code chunks.

``CodeRetriever`` does not parse files or talk to FAISS itself. It
calls the pieces that already exist:

1. ``scan_repository`` — find ``.py`` files (read-only, never imported).
2. ``chunk_repository`` — split them into functions, classes, leftovers.
3. ``EmbeddingGenerator.embed_chunks`` / ``embed_query`` — numbers.
4. ``VectorStore.add`` / ``search`` — FAISS nearest neighbors.

You inject the generator and the store so tests can pass fakes and
never download the MiniLM model.
"""

from __future__ import annotations

import os
from typing import Any

from repopilot_ai.ingestion.code_chunker import chunk_repository
from repopilot_ai.ingestion.repository_scanner import scan_repository


class CodeRetriever:
    """Index a local Python repo, then search it with a question."""

    def __init__(self, embedding_generator: Any, vector_store: Any) -> None:
        if embedding_generator is None:
            raise ValueError("embedding_generator is required")
        if vector_store is None:
            raise ValueError("vector_store is required")
        _require_callable(embedding_generator, "embed_chunks")
        _require_callable(embedding_generator, "embed_query")
        _require_callable(vector_store, "add")
        _require_callable(vector_store, "search")

        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    def index_repository(self, repository_path: str | os.PathLike[str]) -> dict[str, Any]:
        """Scan, chunk, embed, and store one repository.

        Returns a summary dict. A missing folder is ``success=False``.
        An empty folder is ``success=True`` with zero indexed chunks.
        """
        if not isinstance(repository_path, (str, os.PathLike)):
            raise TypeError("repository_path must be a path string")

        scan_result = scan_repository(repository_path)
        if not scan_result.get("success", False):
            return {
                "success": False,
                "error": scan_result.get("error") or "Repository scan failed",
                "repository_path": scan_result.get("repository_path"),
                "total_discovered_files": 0,
                "total_chunks": 0,
                "indexed_chunks": 0,
            }

        chunk_result = chunk_repository(scan_result)
        chunks = list(chunk_result.get("chunks") or [])
        if not chunks:
            return _index_summary(
                scan_result=scan_result,
                total_chunks=0,
                indexed_chunks=0,
            )

        embedded = self.embedding_generator.embed_chunks(chunks)
        added = self.vector_store.add(embedded)
        return _index_summary(
            scan_result=scan_result,
            total_chunks=int(chunk_result.get("total_chunks") or len(chunks)),
            indexed_chunks=int(added),
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Embed ``query`` and return the closest stored chunks.

        Each item is ``{"score": float, "chunk": {...metadata...}}``.
        An empty index returns ``[]``. Invalid queries raise.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if query.strip() == "":
            raise ValueError("query must not be empty")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if getattr(self.vector_store, "is_empty", False) or len(self.vector_store) == 0:
            return []

        query_embedding = self.embedding_generator.embed_query(query)
        return self.vector_store.search(query_embedding, k=top_k)


def _index_summary(
    scan_result: dict[str, Any],
    total_chunks: int,
    indexed_chunks: int,
) -> dict[str, Any]:
    return {
        "success": True,
        "error": None,
        "repository_path": scan_result.get("repository_path"),
        "total_discovered_files": int(scan_result.get("total_discovered_files") or 0),
        "total_chunks": total_chunks,
        "indexed_chunks": indexed_chunks,
    }


def _require_callable(obj: Any, name: str) -> None:
    if not callable(getattr(obj, name, None)):
        raise TypeError(f"{type(obj).__name__} is missing callable {name}()")
