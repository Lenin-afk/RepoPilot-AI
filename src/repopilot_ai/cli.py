"""Command-line interface for asking questions about a Python repository."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from repopilot_ai.llm_providers import OllamaProvider
from repopilot_ai.retrieval import (
    CodeRetriever,
    EmbeddingGenerator,
    RAGPipeline,
    VectorStore,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the RepoPilot command-line argument parser."""
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Ask questions about a local Python repository."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=project_root,
        help="repository folder to index (default: RepoPilot AI itself)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("REPOPILOT_OLLAMA_MODEL", "qwen2.5-coder:1.5b"),
        help="installed Ollama model (default: qwen2.5-coder:1.5b)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="number of code chunks to include as context (default: 5)",
    )
    return parser


def main() -> None:
    """Index a repository, then answer questions until the user exits."""
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be a positive integer")

    repository_path = args.repo.expanduser().resolve()
    print(f"Loading the embedding model and indexing {repository_path}...")
    retriever = CodeRetriever(EmbeddingGenerator(), VectorStore())
    indexed = retriever.index_repository(repository_path)
    if not indexed["success"] or indexed["indexed_chunks"] == 0:
        print(f"Indexing failed: {indexed.get('error') or 'no Python code found'}")
        return
    print(
        f"Indexed {indexed['indexed_chunks']} chunks from "
        f"{indexed['total_discovered_files']} Python files."
    )

    rag = RAGPipeline(retriever)
    provider = OllamaProvider(model=args.model)
    print("\nAsk a question about the code. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = rag.prepare(question, top_k=args.top_k)
        print("\nRetrieved sources:")
        if not result["sources"]:
            print("  No relevant code chunks were found.")
        for source in result["sources"]:
            score = source["score"]
            score_text = f" | vector similarity {score:.3f}" if score is not None else ""
            print(
                f"  - {source['citation']} "
                f"({source['chunk_type']} {source['symbol_name']}){score_text}"
            )

        try:
            answer = provider.generate(result["prompt"])
        except RuntimeError as exc:
            print(f"\nNo model answer: {exc}")
            print("\nPrompt prepared for the local model provider:\n")
            print(result["prompt"])
        else:
            print(f"\nRepoPilot answer:\n{answer}")
            verified_terms = ("0.75", "0.12", "-0.5", "-0.3", "not full TF-IDF")
            if result.get("verified_note") and not all(
                term in answer for term in verified_terms
            ):
                print(f"\nVerified from code:\n{result['verified_note']}")
