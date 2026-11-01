"""Check whether common RepoPilot questions retrieve the expected source files."""

from __future__ import annotations

import argparse
from pathlib import Path

from repopilot_ai.retrieval import (
    CodeRetriever,
    EmbeddingGenerator,
    RAGPipeline,
    VectorStore,
)


EVALUATION_CASES = (
    (
        "What does this project do?",
        "src/repopilot_ai/__init__.py",
    ),
    (
        "How does the scanner find Python files and skip folders?",
        "src/repopilot_ai/ingestion/repository_scanner.py",
    ),
    (
        "How does the parser extract classes and functions?",
        "src/repopilot_ai/ingestion/python_parser.py",
    ),
    (
        "How are Python files split into code chunks?",
        "src/repopilot_ai/ingestion/code_chunker.py",
    ),
    (
        "How does a question become an embedding?",
        "src/repopilot_ai/retrieval/embedding_generator.py",
    ),
    (
        "How does FAISS choose the nearest code chunks?",
        "src/repopilot_ai/retrieval/vector_store.py",
    ),
    (
        "How is repository code turned into a prompt for an LLM?",
        "src/repopilot_ai/retrieval/rag.py",
    ),
    (
        "How does RepoPilot call the local model?",
        "src/repopilot_ai/llm_providers.py",
    ),
)


def main() -> int:
    """Index a repository and report expected-source recall at five."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=project_root,
        help="RepoPilot-AI checkout to check (default: this project)",
    )
    args = parser.parse_args()

    retriever = CodeRetriever(EmbeddingGenerator(), VectorStore())
    indexed = retriever.index_repository(args.repo.expanduser().resolve())
    if not indexed["success"] or indexed["indexed_chunks"] == 0:
        print(f"Indexing failed: {indexed.get('error') or 'no Python code found'}")
        return 1

    pipeline = RAGPipeline(retriever)
    passed = 0
    print(
        f"Checking {len(EVALUATION_CASES)} questions against "
        f"{indexed['indexed_chunks']} chunks...\n"
    )
    for question, expected_path in EVALUATION_CASES:
        result = pipeline.prepare(question, top_k=5)
        source_paths = [source["file_path"] for source in result["sources"]]
        found = expected_path in source_paths
        passed += int(found)
        marker = "PASS" if found else "MISS"
        print(f"{marker} {question}")
        if not found:
            print(f"  Expected in top 5: {expected_path}")
            print(f"  Retrieved: {', '.join(source_paths) or 'no sources'}")

    print(f"\nRetrieval check: {passed}/{len(EVALUATION_CASES)} passed.")
    return 0 if passed == len(EVALUATION_CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
