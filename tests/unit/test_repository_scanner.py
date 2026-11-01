"""Unit tests for the repository file scanner.

Run with:

    python -m unittest tests.unit.test_repository_scanner -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from repopilot_ai.ingestion.python_parser import parse_python_source
from repopilot_ai.ingestion.repository_scanner import scan_repository


class TestRepositoryScanner(unittest.TestCase):
    """Checks for scan_repository()."""

    def test_multiple_python_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
            (root / "beta.py").write_text("def beta():\n    return 2\n", encoding="utf-8")

            result = scan_repository(root)

            self.assertTrue(result["success"])
            self.assertIsNone(result["error"])
            self.assertEqual(result["total_discovered_files"], 2)
            self.assertEqual(result["successfully_parsed_files"], 2)
            self.assertEqual(result["failed_files"], 0)
            self.assertEqual(
                [item["path"] for item in result["files"]],
                ["alpha.py", "beta.py"],
            )

    def test_nested_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "pkg" / "inner"
            nested.mkdir(parents=True)
            (root / "top.py").write_text("def top():\n    pass\n", encoding="utf-8")
            (nested / "leaf.py").write_text("def leaf():\n    pass\n", encoding="utf-8")

            result = scan_repository(root)

            paths = [item["path"] for item in result["files"]]
            self.assertEqual(paths, ["pkg/inner/leaf.py", "top.py"])
            self.assertEqual(result["total_discovered_files"], 2)

    def test_ignored_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("def app():\n    pass\n", encoding="utf-8")

            ignored = [
                ".git",
                ".venv",
                "venv",
                "__pycache__",
                "node_modules",
                ".pytest_cache",
                "dist",
                "build",
                ".hidden",
            ]
            for name in ignored:
                folder = root / name
                folder.mkdir()
                (folder / "skip_me.py").write_text("def skip():\n    pass\n", encoding="utf-8")

            result = scan_repository(root)

            self.assertEqual(result["total_discovered_files"], 1)
            self.assertEqual(result["files"][0]["path"], "src/app.py")

    def test_empty_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = scan_repository(temp_dir)

            self.assertTrue(result["success"])
            self.assertIsNone(result["error"])
            self.assertEqual(result["total_discovered_files"], 0)
            self.assertEqual(result["successfully_parsed_files"], 0)
            self.assertEqual(result["failed_files"], 0)
            self.assertEqual(result["files"], [])

    def test_missing_path(self) -> None:
        missing = Path(tempfile.gettempdir()) / "repopilot-missing-repo-does-not-exist"
        result = scan_repository(missing)

        self.assertFalse(result["success"])
        self.assertIn("does not exist", result["error"])
        self.assertEqual(result["total_discovered_files"], 0)
        self.assertEqual(result["files"], [])

    def test_invalid_python_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "good.py").write_text("def good():\n    return 1\n", encoding="utf-8")
            (root / "bad.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

            result = scan_repository(root)

            self.assertTrue(result["success"])
            self.assertEqual(result["total_discovered_files"], 2)
            self.assertEqual(result["successfully_parsed_files"], 1)
            self.assertEqual(result["failed_files"], 1)

            by_path = {item["path"]: item for item in result["files"]}
            self.assertTrue(by_path["good.py"]["success"])
            self.assertFalse(by_path["bad.py"]["success"])
            self.assertTrue(by_path["bad.py"]["error"])

    def test_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pkg").mkdir()
            (root / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")

            result = scan_repository(root)
            file_path = result["files"][0]["path"]

            self.assertEqual(file_path, "pkg/mod.py")
            self.assertFalse(file_path.startswith("/"))
            self.assertFalse(Path(file_path).is_absolute())
            self.assertIn("pkg", file_path)
            self.assertNotIn(str(root), file_path)

    def test_ast_parser_integration(self) -> None:
        source = '''def greet(name):
    """Say hello."""
    return name

class Greeter:
    """Wraps greet."""

    def run(self, name):
        """Call greet."""
        return greet(name)
'''
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "greet.py").write_text(source, encoding="utf-8")

            result = scan_repository(root)
            parsed = parse_python_source(source)
            scanned = result["files"][0]

            self.assertTrue(result["success"])
            self.assertTrue(scanned["success"])
            self.assertEqual(scanned["functions"], parsed["functions"])
            self.assertEqual(scanned["classes"], parsed["classes"])
            self.assertEqual(scanned["functions"][0]["name"], "greet")
            self.assertEqual(scanned["classes"][0]["name"], "Greeter")
            self.assertEqual(scanned["classes"][0]["methods"][0]["name"], "run")


if __name__ == "__main__":
    unittest.main()
