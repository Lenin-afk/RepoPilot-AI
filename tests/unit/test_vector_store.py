"""Unit tests for the FAISS vector store.

These tests use small hand-made (normalized) vectors so they do not
load the Sentence Transformer. Run with:

    python -m unittest tests.unit.test_vector_store -v
"""

from __future__ import annotations

import unittest

import numpy as np

from repopilot_ai.retrieval.embedding_generator import DEFAULT_EMBEDDING_DIMENSION
from repopilot_ai.retrieval.vector_store import VectorStore


def _normalize(values: list[float], dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> np.ndarray:
    """Pad a short vector with zeros, then scale it to length 1."""
    if len(values) > dimension:
        raise ValueError("test vector is longer than the index dimension")
    padded = np.zeros(dimension, dtype=np.float32)
    padded[: len(values)] = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(padded))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return padded / np.float32(norm)


def _chunk(embedding: np.ndarray, **overrides: object) -> dict:
    chunk = {
        "chunk_id": "mod.py:function:add:1-3",
        "file_path": "mod.py",
        "symbol_name": "add",
        "chunk_type": "function",
        "start_line": 1,
        "end_line": 3,
        "content": "def add(a, b):\n    return a + b\n",
        "embedding": embedding,
    }
    chunk.update(overrides)
    return chunk


class TestVectorStore(unittest.TestCase):
    """Checks for VectorStore."""

    def test_starts_empty(self) -> None:
        store = VectorStore()

        self.assertEqual(store.dimension, DEFAULT_EMBEDDING_DIMENSION)
        self.assertEqual(len(store), 0)
        self.assertTrue(store.is_empty)
        self.assertEqual(store.search(_normalize([1.0, 0.0]), k=3), [])

    def test_empty_add_is_a_no_op(self) -> None:
        store = VectorStore()

        added = store.add([])

        self.assertEqual(added, 0)
        self.assertEqual(len(store), 0)

    def test_add_and_search_nearest_neighbor(self) -> None:
        add_vec = _normalize([1.0, 0.0, 0.0])
        sub_vec = _normalize([0.0, 1.0, 0.0])
        store = VectorStore()
        store.add(
            [
                _chunk(add_vec, symbol_name="add", content="def add(a, b): return a + b"),
                _chunk(sub_vec, symbol_name="subtract", content="def subtract(a, b): return a - b"),
            ]
        )

        hits = store.search(add_vec, k=1)

        self.assertEqual(len(store), 2)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["chunk"]["symbol_name"], "add")
        self.assertAlmostEqual(hits[0]["score"], 1.0, places=5)

    def test_top_k_order(self) -> None:
        first = _normalize([1.0, 0.0, 0.0])
        second = _normalize([0.8, 0.2, 0.0])
        third = _normalize([0.0, 1.0, 0.0])
        store = VectorStore()
        store.add(
            [
                _chunk(first, symbol_name="first"),
                _chunk(second, symbol_name="second"),
                _chunk(third, symbol_name="third"),
            ]
        )

        hits = store.search(first, k=2)
        names = [hit["chunk"]["symbol_name"] for hit in hits]

        self.assertEqual(names, ["first", "second"])
        self.assertGreaterEqual(hits[0]["score"], hits[1]["score"])

    def test_k_larger_than_index_returns_all(self) -> None:
        store = VectorStore()
        store.add(
            [
                _chunk(_normalize([1.0, 0.0]), symbol_name="a"),
                _chunk(_normalize([0.0, 1.0]), symbol_name="b"),
            ]
        )

        hits = store.search(_normalize([1.0, 0.0]), k=50)

        self.assertEqual(len(hits), 2)
        self.assertTrue(all(hit["chunk"]["symbol_name"] in {"a", "b"} for hit in hits))

    def test_metadata_and_source_are_preserved(self) -> None:
        embedding = _normalize([0.3, 0.4, 0.5])
        original = _chunk(
            embedding,
            extra_field="keep-me",
            content="def add(a, b):\n    return a + b\n",
        )
        store = VectorStore()
        store.add([original])

        hits = store.search(embedding, k=1)
        chunk = hits[0]["chunk"]

        self.assertEqual(chunk["chunk_id"], original["chunk_id"])
        self.assertEqual(chunk["file_path"], "mod.py")
        self.assertEqual(chunk["symbol_name"], "add")
        self.assertEqual(chunk["chunk_type"], "function")
        self.assertEqual(chunk["start_line"], 1)
        self.assertEqual(chunk["end_line"], 3)
        self.assertEqual(chunk["content"], original["content"])
        self.assertEqual(chunk["extra_field"], "keep-me")
        self.assertNotIn("embedding", chunk)
        # Caller dict is not mutated by add().
        self.assertIn("embedding", original)

    def test_incremental_add(self) -> None:
        store = VectorStore()
        store.add([_chunk(_normalize([1.0, 0.0]), symbol_name="one")])
        store.add([_chunk(_normalize([0.0, 1.0]), symbol_name="two")])

        hits = store.search(_normalize([0.0, 1.0]), k=1)

        self.assertEqual(len(store), 2)
        self.assertEqual(hits[0]["chunk"]["symbol_name"], "two")

    def test_wrong_embedding_dimension_raises(self) -> None:
        store = VectorStore(dimension=DEFAULT_EMBEDDING_DIMENSION)
        short = np.ones(8, dtype=np.float32)

        with self.assertRaises(ValueError):
            store.add([_chunk(short)])
        with self.assertRaises(ValueError):
            store.search(short, k=1)

    def test_missing_embedding_raises_instead_of_dropping(self) -> None:
        store = VectorStore()
        payload = _chunk(_normalize([1.0, 0.0]))
        del payload["embedding"]

        with self.assertRaises(ValueError):
            store.add([payload])
        self.assertEqual(len(store), 0)

    def test_invalid_inputs_raise(self) -> None:
        store = VectorStore()
        good = _chunk(_normalize([1.0, 0.0]))

        with self.assertRaises(ValueError):
            VectorStore(dimension=0)
        with self.assertRaises(TypeError):
            store.add("not a list")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            store.add([{"embedding": "nope"}])
        with self.assertRaises(ValueError):
            store.add([good])
            store.search(_normalize([1.0, 0.0]), k=0)
        with self.assertRaises(ValueError):
            store.search(np.array([np.nan] * DEFAULT_EMBEDDING_DIMENSION), k=1)

    def test_custom_dimension(self) -> None:
        store = VectorStore(dimension=4)
        vector = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        vector = vector / np.linalg.norm(vector)
        store.add(
            [
                {
                    "symbol_name": "tiny",
                    "content": "x = 1",
                    "embedding": vector,
                }
            ]
        )

        hits = store.search(vector, k=1)

        self.assertEqual(store.dimension, 4)
        self.assertEqual(hits[0]["chunk"]["content"], "x = 1")
        self.assertAlmostEqual(hits[0]["score"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
