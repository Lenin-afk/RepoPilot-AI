"""Code ingestion helpers for RepoPilot AI.

This package turns source files into structured data that later
parts of the app (search, Q&A, etc.) can use.
"""

from repopilot_ai.ingestion.python_parser import parse_python_source

__all__ = ["parse_python_source"]
