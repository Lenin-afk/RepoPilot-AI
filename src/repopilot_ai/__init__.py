"""RepoPilot AI answers questions about local Python repositories.

It scans and parses Python files, splits code into chunks, embeds the chunks,
and uses FAISS to retrieve relevant code for a grounded prompt. The command
line can send that prompt to a local Ollama model. This package has no web
interface or database.
"""

from repopilot_ai.cli import main

__all__ = ["main"]
