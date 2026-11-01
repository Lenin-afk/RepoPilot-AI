"""Unit tests for CodeRetriever.

The real MiniLM model is never loaded. A small fake generator turns
text into vectors so FAISS can still rank chunks. Run with:

    pytest tests/unit/test_retriever.py -v
"""

from __future__ import annotations

import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from repopilot_ai.retrieval.retriever import CodeRetriever
from repopilot_ai.retrieval.vector_store import VectorStore

FAKE_DIMENSION = 8

ADD_SOURCE = '''def add(a, b):
    """Add two numbers."""
    return a + b
'''

MULTIPLY_SOURCE = '''def multiply(a, b):
    """Multiply two numbers."""
    return a * b
'''


class FakeEmbeddingGenerator:
    """Deterministic embeddings from token counts. No model download."""

    def __init__(self, dimension: int = FAKE_DIMENSION) -> None:
        self.embedding_dimension = dimension

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        embedded = []
        for chunk in chunks:
            item = dict(chunk)
            item["embedding"] = self.embed_query(str(chunk.get("content", "")))
            embedded.append(item)
        return embedded

    def embed_query(self, query: str) -> np.ndarray:
        vector = np.zeros(self.embedding_dimension, dtype=np.float32)
        tokens = (
            query.lower()
            .replace("(", " ")
            .replace(")", " ")
            .replace(":", " ")
            .replace('"', " ")
            .split()
        )
        if not tokens:
            vector[0] = 1.0
            return vector
        for token in tokens:
            bucket = zlib.crc32(token.encode("utf-8")) % self.embedding_dimension
            vector[bucket] += 1.0
        return vector / np.linalg.norm(vector)


class TestCodeRetriever(unittest.TestCase):
    """Checks for CodeRetriever using a fake embedder and temp repos."""

    def _retriever(self) -> CodeRetriever:
        generator = FakeEmbeddingGenerator()
        store = VectorStore(dimension=FAKE_DIMENSION)
        return CodeRetriever(generator, store)

    def _write_repo(self, root: Path) -> None:
        (root / "math_add.py").write_text(ADD_SOURCE, encoding="utf-8")
        (root / "math_mul.py").write_text(MULTIPLY_SOURCE, encoding="utf-8")

    def test_index_and_search_returns_relevant_chunk(self) -> None:
        retriever = self._retriever()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_repo(root)

            summary = retriever.index_repository(root)
            hits = retriever.search("How does add two numbers work?", top_k=1)

        self.assertTrue(summary["success"])
        self.assertGreaterEqual(summary["indexed_chunks"], 2)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["chunk"]["symbol_name"], "add")
        self.assertIn("def add", hits[0]["chunk"]["content"])
        self.assertIn("score", hits[0])
        self.assertGreater(hits[0]["score"], 0.0)

    def test_search_preserves_metadata(self) -> None:
        retriever = self._retriever()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_repo(root)
            retriever.index_repository(root)
            hits = retriever.search("multiply two numbers", top_k=5)

        self.assertGreaterEqual(len(hits), 1)
        multiply = next(hit for hit in hits if hit["chunk"]["symbol_name"] == "multiply")
        chunk = multiply["chunk"]
        self.assertEqual(chunk["file_path"], "math_mul.py")
        self.assertEqual(chunk["chunk_type"], "function")
        self.assertIn("content", chunk)
        self.assertIn("chunk_id", chunk)
        self.assertNotIn("embedding", chunk)

    def test_top_k_limits_results(self) -> None:
        retriever = self._retriever()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_repo(root)
            retriever.index_repository(root)
            hits = retriever.search("numbers", top_k=1)

        self.assertEqual(len(hits), 1)

    def test_empty_repository(self) -> None:
        retriever = self._retriever()
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = retriever.index_repository(temp_dir)
            hits = retriever.search("anything at all", top_k=5)

        self.assertTrue(summary["success"])
        self.assertEqual(summary["total_discovered_files"], 0)
        self.assertEqual(summary["total_chunks"], 0)
        self.assertEqual(summary["indexed_chunks"], 0)
        self.assertEqual(hits, [])

    def test_missing_repository(self) -> None:
        retriever = self._retriever()
        missing = Path(tempfile.gettempdir()) / "repopilot-does-not-exist-xyz"

        summary = retriever.index_repository(missing)

        self.assertFalse(summary["success"])
        self.assertIsNotNone(summary["error"])
        self.assertEqual(summary["indexed_chunks"], 0)
        self.assertEqual(retriever.search("add numbers", top_k=3), [])

    def test_search_before_index_is_empty(self) -> None:
        retriever = self._retriever()

        self.assertEqual(retriever.search("add numbers", top_k=3), [])

    def test_invalid_query_raises(self) -> None:
        retriever = self._retriever()

        with self.assertRaises(TypeError):
            retriever.search(None, top_k=5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            retriever.search("   ", top_k=5)
        with self.assertRaises(ValueError):
            retriever.search("add", top_k=0)

    def test_constructor_rejects_incomplete_helpers(self) -> None:
        store = VectorStore(dimension=FAKE_DIMENSION)

        with self.assertRaises(ValueError):
            CodeRetriever(None, store)
        with self.assertRaises(TypeError):
            CodeRetriever(object(), store)


if __name__ == "__main__":
    unittest.main()
