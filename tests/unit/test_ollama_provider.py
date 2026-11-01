"""Tests for the local Ollama text-generation adapter."""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from repopilot_ai.llm_providers import OllamaProvider


class TestOllamaProvider(unittest.TestCase):
    @patch("repopilot_ai.llm_providers.urlopen")
    def test_posts_prompt_and_returns_response(self, urlopen: MagicMock) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"response":"  The answer.  "}'
        urlopen.return_value = response

        provider = OllamaProvider(model="test-model")
        answer = provider.generate("Explain this code")

        self.assertEqual(answer, "The answer.")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/generate")
        self.assertEqual(
            json.loads(request.data),
            {
                "model": "test-model",
                "prompt": "Explain this code",
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 120.0)

    @patch("repopilot_ai.llm_providers.urlopen", side_effect=URLError("refused"))
    def test_unavailable_server_has_clear_error(self, _urlopen: MagicMock) -> None:
        with self.assertRaisesRegex(RuntimeError, "Could not reach Ollama"):
            OllamaProvider().generate("hello")

    @patch(
        "repopilot_ai.llm_providers.urlopen",
        side_effect=HTTPError(
            "http://localhost:11434/api/generate",
            404,
            "not found",
            {},
            io.BytesIO(b"model not found"),
        ),
    )
    def test_missing_model_has_clear_error(self, _urlopen: MagicMock) -> None:
        with self.assertRaisesRegex(RuntimeError, "could not find model"):
            OllamaProvider(model="missing").generate("hello")

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OllamaProvider(model=" ")
        with self.assertRaises(ValueError):
            OllamaProvider(timeout=0)
        with self.assertRaises(ValueError):
            OllamaProvider().generate(" ")
        with self.assertRaises(TypeError):
            OllamaProvider().generate(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
