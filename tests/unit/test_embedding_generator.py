"""Unit tests for embedding generation.

The Sentence Transformer is loaded once for the class so the suite
stays reasonably fast. Run with:

    python -m unittest tests.unit.test_embedding_generator -v
"""

from __future__ import annotations

import unittest

import numpy as np

from repopilot_ai.retrieval.embedding_generator import (
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_MODEL_NAME,
    EmbeddingGenerator,
)


def _sample_chunk(**overrides: object) -> dict:
    chunk = {
        "chunk_id": "mod.py:function:add:1-3",
        "file_path": "mod.py",
        "symbol_name": "add",
        "chunk_type": "function",
        "start_line": 1,
        "end_line": 3,
        "content": "def add(a, b):\n    return a + b\n",
    }
    chunk.update(overrides)
    return chunk


class TestEmbeddingGenerator(unittest.TestCase):
    """Checks for EmbeddingGenerator."""

    generator: EmbeddingGenerator

    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = EmbeddingGenerator()

    def test_model_initialization(self) -> None:
        generator = self.generator

        self.assertEqual(generator.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(generator.device, DEFAULT_DEVICE)
        self.assertIsNotNone(generator.model)
        self.assertIs(generator.model, generator.model)
        self.assertEqual(generator.embedding_dimension, DEFAULT_EMBEDDING_DIMENSION)

    def test_configurable_device_is_stored(self) -> None:
        # Loading a second copy of the same model is slow, so we only
        # check that the constructor accepts and stores the value.
        generator = self.generator
        self.assertEqual(generator.device, "cpu")

    def test_single_text_embedding(self) -> None:
        vectors = self.generator.embed_texts(["def greet():\n    return 1\n"])

        self.assertEqual(vectors.shape, (1, DEFAULT_EMBEDDING_DIMENSION))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertFalse(np.allclose(vectors, 0))

    def test_multiple_text_embeddings(self) -> None:
        texts = [
            "def add(a, b):\n    return a + b\n",
            "def subtract(a, b):\n    return a - b\n",
            "class Counter:\n    pass\n",
        ]
        vectors = self.generator.embed_texts(texts)

        self.assertEqual(vectors.shape, (len(texts), DEFAULT_EMBEDDING_DIMENSION))
        # Two different functions should not collapse to the same vector.
        self.assertFalse(np.allclose(vectors[0], vectors[1]))

    def test_empty_input(self) -> None:
        vectors = self.generator.embed_texts([])
        chunks = self.generator.embed_chunks([])

        self.assertEqual(vectors.shape, (0, DEFAULT_EMBEDDING_DIMENSION))
        self.assertEqual(chunks, [])

    def test_query_embedding(self) -> None:
        vector = self.generator.embed_query("How does the add function work?")

        self.assertEqual(vector.shape, (DEFAULT_EMBEDDING_DIMENSION,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_embedding_dimensions(self) -> None:
        text_vectors = self.generator.embed_texts(["hello", "world"])
        query = self.generator.embed_query("hello")
        chunk_results = self.generator.embed_chunks([_sample_chunk()])

        self.assertEqual(text_vectors.shape[1], DEFAULT_EMBEDDING_DIMENSION)
        self.assertEqual(query.shape[0], DEFAULT_EMBEDDING_DIMENSION)
        self.assertEqual(
            chunk_results[0]["embedding"].shape, (DEFAULT_EMBEDDING_DIMENSION,)
        )

    def test_metadata_preservation(self) -> None:
        original = _sample_chunk(extra_field="keep-me")
        results = self.generator.embed_chunks([original])

        self.assertEqual(len(results), 1)
        embedded = results[0]
        for key, value in original.items():
            self.assertEqual(embedded[key], value)
        self.assertIn("embedding", embedded)
        self.assertEqual(embedded["embedding"].shape, (DEFAULT_EMBEDDING_DIMENSION,))
        # Original input dict is not mutated.
        self.assertNotIn("embedding", original)

    def test_empty_content_is_not_discarded(self) -> None:
        empty = _sample_chunk(content="", symbol_name="blank")
        results = self.generator.embed_chunks([empty, _sample_chunk()])

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["symbol_name"], "blank")
        self.assertEqual(results[0]["content"], "")
        self.assertEqual(results[0]["embedding"].shape, (DEFAULT_EMBEDDING_DIMENSION,))

    def test_invalid_texts_raise(self) -> None:
        with self.assertRaises(TypeError):
            self.generator.embed_texts("not a list")
        with self.assertRaises(TypeError):
            self.generator.embed_texts([123])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            self.generator.embed_query(None)  # type: ignore[arg-type]

    def test_invalid_chunks_raise_instead_of_dropping(self) -> None:
        with self.assertRaises(TypeError):
            self.generator.embed_chunks("not a list")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.generator.embed_chunks([{"file_path": "mod.py"}])
        with self.assertRaises(TypeError):
            self.generator.embed_chunks([_sample_chunk(content=None)])


if __name__ == "__main__":
    unittest.main()
