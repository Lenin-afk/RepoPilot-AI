"""Unit tests for semantic code chunking.

Run with:

    python -m unittest tests.unit.test_code_chunker -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from repopilot_ai.ingestion.code_chunker import chunk_repository
from repopilot_ai.ingestion.repository_scanner import scan_repository


MULTI_FUNCTION_SOURCE = '''import os

MAX_ITEMS = 10

def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract two numbers."""
    return a - b
'''

CLASS_SOURCE = '''"""Module docstring."""

from typing import Any

FLAG = True

class Counter:
    """Counts things."""

    def __init__(self, start):
        """Create a counter."""
        self.value = start

    def increment(self):
        """Increase the value."""
        self.value += 1

def helper():
    return FLAG
'''


class TestCodeChunker(unittest.TestCase):
    """Checks for chunk_repository()."""

    def _chunk_source(self, source: str, filename: str = "mod.py", **kwargs) -> dict:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / filename).write_text(source, encoding="utf-8")
            scan_result = scan_repository(root)
            return chunk_repository(scan_result, **kwargs)

    def test_multiple_functions(self) -> None:
        result = self._chunk_source(MULTI_FUNCTION_SOURCE)
        functions = [c for c in result["chunks"] if c["chunk_type"] == "function"]
        names = [c["symbol_name"] for c in functions]

        self.assertTrue(result["success"])
        self.assertEqual(names, ["add", "subtract"])
        self.assertIn("def add(a, b):", functions[0]["content"])
        self.assertIn("def subtract(a, b):", functions[1]["content"])

    def test_classes_with_methods(self) -> None:
        result = self._chunk_source(CLASS_SOURCE)
        class_chunks = [c for c in result["chunks"] if c["chunk_type"] == "class"]
        function_chunks = [c for c in result["chunks"] if c["chunk_type"] == "function"]
        function_names = [c["symbol_name"] for c in function_chunks]

        self.assertEqual(len(class_chunks), 1)
        self.assertEqual(class_chunks[0]["symbol_name"], "Counter")
        self.assertIn("def __init__(self, start):", class_chunks[0]["content"])
        self.assertIn("def increment(self):", class_chunks[0]["content"])
        # Methods must not appear as extra function chunks.
        self.assertEqual(function_names, ["helper"])
        self.assertNotIn("__init__", function_names)
        self.assertNotIn("increment", function_names)

    def test_module_level_imports_and_constants(self) -> None:
        result = self._chunk_source(MULTI_FUNCTION_SOURCE)
        file_chunks = [c for c in result["chunks"] if c["chunk_type"] == "file"]

        self.assertEqual(len(file_chunks), 1)
        content = file_chunks[0]["content"]
        self.assertIn("import os", content)
        self.assertIn("MAX_ITEMS = 10", content)
        self.assertNotIn("def add", content)
        self.assertNotIn("def subtract", content)
        self.assertEqual(file_chunks[0]["symbol_name"], "__module__")

    def test_empty_files(self) -> None:
        result = self._chunk_source("")

        self.assertTrue(result["success"])
        self.assertEqual(result["total_chunks"], 0)
        self.assertEqual(result["chunks"], [])

    def test_invalid_source(self) -> None:
        result = self._chunk_source("def broken(:\n    pass\n", filename="bad.py")
        file_chunks = [c for c in result["chunks"] if c["chunk_type"] == "file"]

        self.assertTrue(result["success"])
        self.assertEqual(len(file_chunks), 1)
        self.assertIn("def broken", file_chunks[0]["content"])
        self.assertEqual(
            [c for c in result["chunks"] if c["chunk_type"] in {"function", "class"}],
            [],
        )

    def test_metadata(self) -> None:
        result = self._chunk_source(MULTI_FUNCTION_SOURCE)
        required = {
            "chunk_id",
            "file_path",
            "symbol_name",
            "chunk_type",
            "start_line",
            "end_line",
            "content",
        }

        self.assertGreater(result["total_chunks"], 0)
        for chunk in result["chunks"]:
            self.assertEqual(set(chunk.keys()), required)
            self.assertEqual(chunk["file_path"], "mod.py")
            self.assertIsInstance(chunk["chunk_id"], str)
            self.assertTrue(chunk["chunk_id"])
            self.assertGreaterEqual(chunk["end_line"], chunk["start_line"])

        add = next(c for c in result["chunks"] if c["symbol_name"] == "add")
        self.assertEqual(add["start_line"], 5)
        self.assertEqual(add["end_line"], 7)

    def test_maximum_size(self) -> None:
        result = self._chunk_source(MULTI_FUNCTION_SOURCE, max_chunk_size=40)
        add_parts = [c for c in result["chunks"] if c["symbol_name"] == "add"]

        self.assertGreater(len(add_parts), 1)
        for part in add_parts:
            # A part may exceed the limit only when a single line is longer
            # than max_chunk_size. These lines are short, so each part fits.
            self.assertLessEqual(len(part["content"]), 40)

    def test_oversized_functions(self) -> None:
        long_line = "    return " + ("x" * 80)
        source = f"def huge():\n{long_line}\n"
        result = self._chunk_source(source, max_chunk_size=20)
        parts = [c for c in result["chunks"] if c["symbol_name"] == "huge"]

        self.assertGreater(len(parts), 1)
        # The oversized line is kept in full; we do not cut it.
        self.assertTrue(any(long_line in part["content"] for part in parts))
        self.assertTrue(all(part["content"] != "" for part in parts))

    def test_source_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mod.py").write_text(CLASS_SOURCE, encoding="utf-8")
            scan_result = scan_repository(root)
            parsed_file = scan_result["files"][0]
            result = chunk_repository(scan_result, max_chunk_size=50)

            helper_source = parsed_file["functions"][0]["source"]
            class_source = parsed_file["classes"][0]["source"]
            helper_joined = "".join(
                c["content"] for c in result["chunks"] if c["symbol_name"] == "helper"
            )
            class_joined = "".join(
                c["content"] for c in result["chunks"] if c["symbol_name"] == "Counter"
            )

            self.assertEqual(helper_joined, helper_source)
            self.assertEqual(class_joined, class_source)
            for chunk in result["chunks"]:
                self.assertIsInstance(chunk["content"], str)


if __name__ == "__main__":
    unittest.main()
