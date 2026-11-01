"""Tests for the RepoPilot command-line options."""

from __future__ import annotations

import unittest
from pathlib import Path

from repopilot_ai.cli import build_parser


class TestCLI(unittest.TestCase):
    def test_defaults_point_to_repo_and_local_model(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.repo, Path(__file__).resolve().parents[2])
        self.assertEqual(args.model, "qwen2.5-coder:1.5b")
        self.assertEqual(args.top_k, 5)

    def test_options_can_select_a_different_repo_and_model(self) -> None:
        args = build_parser().parse_args(
            ["--repo", "C:/projects/sample", "--model", "local-coder", "--top-k", "8"]
        )

        self.assertEqual(args.repo, Path("C:/projects/sample"))
        self.assertEqual(args.model, "local-coder")
        self.assertEqual(args.top_k, 8)


if __name__ == "__main__":
    unittest.main()
