from pathlib import Path

from repopilot_ai.retrieval.embedding_generator import EmbeddingGenerator
from repopilot_ai.retrieval.vector_store import VectorStore
from repopilot_ai.retrieval.retriever import CodeRetriever


def main():
    project_root = Path(__file__).resolve().parent.parent
    source_path = project_root / "src" / "repopilot_ai"

    print("\nLoading embedding model...")
    generator = EmbeddingGenerator()
    store = VectorStore()

    retriever = CodeRetriever(generator, store)

    print("\nIndexing RepoPilot AI source code...")
    result = retriever.index_repository(source_path)

    print("\nIndexing result:")
    print(result)

    if not result["success"] or result["indexed_chunks"] == 0:
        print("Indexing failed or produced no chunks.")
        return

    questions = [
        "How does the Python AST parser extract functions and classes?",
        "How are code embeddings generated?",
        "How does the vector store search for similar code?",
        "How does the repository scanner find Python files?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        results = retriever.search(question, top_k=3)

        for i, item in enumerate(results, start=1):
            chunk = item["chunk"]
            print(f"\n  Result {i} | Similarity: {item['score']:.4f}")
            print(f"  File: {chunk.get('file_path', 'Unknown')}")
            print(f"  Type: {chunk.get('chunk_type', 'Unknown')}")
            print(f"  Name: {chunk.get('symbol_name', 'Unknown')}")
            print(f"  Code:\n{chunk.get('content', '')[:500]}")


if __name__ == "__main__":
    main()

