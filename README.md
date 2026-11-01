# RepoPilot AI

RepoPilot indexes Python source files and lets you ask questions about a local
repository. It retrieves code with embeddings and FAISS, then can send the
retrieved context to a local Ollama model. No hosted API key is needed.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) for generated answers (optional if you only
  want to prepare retrieval context)

## Set up

From this project folder, install the Python dependencies:

```powershell
uv sync
```

For local answers, open Ollama and download the default model once:

```powershell
ollama pull qwen2.5-coder:1.5b
```

The embedding model is downloaded by Sentence Transformers the first time
RepoPilot runs. Later runs can use its local cache.

## Ask about a repository

By default, RepoPilot indexes this project. To index another Python project,
pass its folder:

```powershell
uv run repopilot-ai --repo "C:\path\to\your\project"
```

Then type a question at `Question>`. Type `exit` to quit. You can choose a
different Ollama model with `--model`, or set `REPOPILOT_OLLAMA_MODEL`.
`--top-k` changes how many retrieved code chunks are given to the model:

```powershell
uv run repopilot-ai --repo "C:\path\to\your\project" --top-k 8
```

If Ollama is unavailable, RepoPilot still prints the retrieved source list
and the prompt it prepared, so retrieval can be inspected without a model.

## How retrieval works

1. The scanner finds Python files and parses them without importing or running
   repository code.
2. The chunker splits files into functions, classes, and module-level code.
3. Sentence Transformers embeds each chunk and the user's question.
4. FAISS compares the question vector with the stored chunk vectors. The
   embedding vectors are normalized, so the inner product used by FAISS is
   cosine similarity. `RAGPipeline` reranks those candidates using both the
   vector similarity and keyword overlap.
5. The pipeline formats the top chunks and their file/line citations into a
   prompt. The Ollama adapter generates an answer from that prompt.

The RAG layer depends only on a retriever and a provider with `generate(prompt)`;
Ollama is one optional provider implementation.

## Tests

```powershell
uv run pytest
```

The tests use local fakes for model generation and do not require an API key or
a running Ollama server.

To check retrieval against eight common RepoPilot questions (this loads the
embedding model but does not call Ollama), run:

```powershell
uv run python scripts/evaluate_retrieval.py
```
