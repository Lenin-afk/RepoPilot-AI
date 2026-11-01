"""Unit tests for the Python AST source parser.

These tests use only the standard library (unittest). Run them with:

    python -m unittest tests.unit.test_python_parser
"""

from __future__ import annotations

import unittest

from repopilot_ai.ingestion.python_parser import parse_python_source


class TestPythonParser(unittest.TestCase):
    """Checks for parse_python_source()."""

    def test_simple_function(self) -> None:
        source = '''def greet(name):
    """Say hello."""
    return f"Hello, {name}"
'''
        result = parse_python_source(source)

        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["functions"]), 1)
        self.assertEqual(result["classes"], [])

        function = result["functions"][0]
        self.assertEqual(function["name"], "greet")
        self.assertEqual(function["kind"], "function")
        self.assertEqual(function["parameters"], ["name"])
        self.assertEqual(function["docstring"], "Say hello.")
        self.assertEqual(function["start_line"], 1)
        self.assertEqual(function["end_line"], 3)
        self.assertIn("def greet(name):", function["source"])
        self.assertIn("Say hello.", function["source"])

    def test_multiple_functions(self) -> None:
        source = '''def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract two numbers."""
    return a - b
'''
        result = parse_python_source(source)

        self.assertTrue(result["success"])
        names = [fn["name"] for fn in result["functions"]]
        self.assertEqual(names, ["add", "subtract"])
        self.assertEqual(result["functions"][0]["start_line"], 1)
        self.assertEqual(result["functions"][1]["start_line"], 5)

    def test_class_with_methods(self) -> None:
        source = '''class Counter:
    """Counts things."""

    def __init__(self, start):
        """Create a counter."""
        self.value = start

    def increment(self):
        """Increase the value."""
        self.value += 1
'''
        result = parse_python_source(source)

        self.assertTrue(result["success"])
        self.assertEqual(result["functions"], [])
        self.assertEqual(len(result["classes"]), 1)

        cls = result["classes"][0]
        self.assertEqual(cls["name"], "Counter")
        self.assertEqual(cls["docstring"], "Counts things.")
        self.assertEqual(cls["start_line"], 1)
        self.assertGreaterEqual(cls["end_line"], cls["start_line"])
        self.assertIn("class Counter:", cls["source"])

        method_names = [method["name"] for method in cls["methods"]]
        self.assertEqual(method_names, ["__init__", "increment"])
        self.assertEqual(cls["methods"][0]["parameters"], ["self", "start"])
        self.assertEqual(cls["methods"][1]["docstring"], "Increase the value.")
        self.assertIn("def increment(self):", cls["methods"][1]["source"])

    def test_parameters(self) -> None:
        source = """def mix(a, b, *args, flag, **kwargs):
    return a
"""
        result = parse_python_source(source)

        self.assertTrue(result["success"])
        self.assertEqual(
            result["functions"][0]["parameters"],
            ["a", "b", "*args", "flag", "**kwargs"],
        )

    def test_missing_docstrings(self) -> None:
        source = """def bare():
    return 1

class Empty:
    def method(self):
        return 2
"""
        result = parse_python_source(source)

        self.assertTrue(result["success"])
        self.assertIsNone(result["functions"][0]["docstring"])
        self.assertIsNone(result["classes"][0]["docstring"])
        self.assertIsNone(result["classes"][0]["methods"][0]["docstring"])

    def test_invalid_syntax(self) -> None:
        source = "def broken(:\n    pass\n"
        result = parse_python_source(source)

        self.assertFalse(result["success"])
        self.assertIsInstance(result["error"], str)
        self.assertTrue(result["error"])
        self.assertEqual(result["functions"], [])
        self.assertEqual(result["classes"], [])

    def test_empty_source_code(self) -> None:
        result = parse_python_source("")

        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["functions"], [])
        self.assertEqual(result["classes"], [])


if __name__ == "__main__":
    unittest.main()
