# Contributing to LocalRAG-Kit

Thank you for your interest in contributing to **LocalRAG-Kit**! We welcome contributions from developers, security researchers, and AI enthusiasts.

---

## 🛠️ Development Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RitualDev-Lab/localrag-kit.git
   cd localrag-kit
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

---

## 🧪 Running Tests

Ensure all unit and integration tests pass before submitting code:
```bash
pytest -v
```

---

## 🏗️ Architecture Overview

LocalRAG-Kit is structured into clean, decoupled layers:

- `localrag/core/`: File harvesting, pathspec ignore parsing, file parsers (Markdown, PDF, Code), and semantic boundary chunkers.
- `localrag/storage/`: SQLite engine with SQLite FTS5 for BM25 keyword search, binary BLOB vector storage, NumPy SIMD cosine similarity, and change detection.
- `localrag/providers/`: Pluggable embedding providers (Fast Offline Hasher, Ollama, OpenAI) and LLM providers (Ollama, OpenAI, Mock).
- `localrag/retrieval/`: Reciprocal Rank Fusion (RRF) hybrid reranking, context synthesizer with token budgeting, and grounded prompt orchestration.
- `localrag/server/`: High-performance FastAPI server with Server-Sent Events (SSE) streaming, REST endpoints, and embedded static web UI.
- `localrag/cli.py`: Interactive CLI with Rich formatting (`scan`, `index`, `search`, `ask`, `stats`, `serve`).

---

## 📝 Contribution Guidelines

1. **Keep it 100% Offline & Zero-Config**:
   Features must never require mandatory external cloud accounts or external database installations (like Chroma, Pinecone, or PostgreSQL). SQLite must remain the zero-infra backbone.
2. **Deterministic Chunker Line Numbers**:
   Every parser and chunker must compute 1-based exact physical start and end line coordinates so citation provenance is verifiable.
3. **Cross-Platform Compatibility**:
   Ensure paths use `pathlib.Path` and console outputs handle Windows CP1252 / UTF-8 safely.

---

## 📜 License

By contributing to LocalRAG-Kit, you agree that your contributions will be licensed under the [MIT License](LICENSE).
