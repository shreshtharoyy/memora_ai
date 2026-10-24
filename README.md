
# Memora AI

**A benchmark for long-term memory in AI agents.**

Memora AI is an engineering and research project focused on evaluating how AI agents store, retrieve, update, and protect long-term memories across conversations.

The goal is to compare different memory strategies using reproducible experiments rather than assuming one approach is always better.

## Objectives

- Evaluate memory retrieval accuracy across conversations.
- Test how agents handle updated and conflicting facts.
- Compare full-history, rolling-summary, and fact-based memory.
- Measure token usage, API cost, and response latency.
- Investigate robustness against memory-poisoning attempts.
- Compare a custom memory implementation with Mem0 OSS.

## Planned Technology Stack

- **Language:** Python 3.12
- **LLM provider:** Groq API
- **Agent orchestration:** LangChain and LangGraph
- **Memory storage:** SQLite and FTS5
- **Semantic retrieval:** Sentence Transformers and MiniLM
- **Hybrid retrieval:** BM25, cosine similarity, and Reciprocal Rank Fusion (RRF)
- **Testing:** pytest
- **Code quality:** Ruff
- **Planned interfaces:** FastAPI and MCP

Technology integrations will be introduced incrementally as development progresses.

## Architecture

The planned architecture separates the agent, memory management, retrieval, and evaluation responsibilities.

```text
Benchmark Runner
       |
       v
Agent Service
       |
       v
Memory Management
       |
       v
Hybrid Retrieval
  |            |
  v            v
SQLite       MiniLM
FTS5         Embeddings
       |
       v
Relevant Memories
       |
       v
Agent Response
       |
       v
Evaluation and Metrics
```

This is the planned logical architecture; service boundaries and deployment details may evolve during implementation.

## Evaluation Metrics

The benchmark is intended to measure:

- Answer accuracy and evidence recall
- Memory update correctness and stale-fact rate
- Input/output token usage and estimated API cost
- Response latency and failure rate
- Memory-poisoning success and sanitizer utility loss

Results and performance claims will be added only after experiments have been run.

## Development Status

**Status: Early development**

The Groq API connection and an initial reusable LLM client have been tested. Automated unit tests, memory storage, hybrid retrieval, benchmark datasets, and the evaluation pipeline remain under development.

## Getting Started

Python 3.12 is used for initial development.

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies as documented during development.
4. Configure the Groq API key using a local `.env` file.
5. Run the available tests and scripts.

Detailed setup instructions will be added as the project stabilizes.

## Security

- Never commit API keys or other secrets.
- Keep `.env` files out of version control.
- Do not include private datasets or sensitive conversation data in public commits.

## License

No license has been selected yet. Until a license is added, do not assume that the repository is available for unrestricted reuse.
