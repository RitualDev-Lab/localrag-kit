# LocalRAG-Kit 🚀

> **100% Offline, Privacy-First Local RAG Search & Chat Engine for Documents and Codebases.**  
> Zero cloud vector database fees. Zero data leaks. Runs entirely on your local machine.

---

## ⚡ Key Highlights

- **🔒 100% Offline & Private**: No data leaves your machine. Compatible with local LLMs (Ollama) and local embeddings.
- **📁 Multi-Format Ingestion**: Scans and parses Markdown (`.md`), Code (`.py`, `.ts`, `.js`, `.go`, `.rs`, `.java`, etc.), Plain Text (`.txt`), HTML, and PDFs.
- **🧩 Structure-Aware Smart Chunking**:
  - **Code**: Splits on function and class declarations while keeping imports and docstrings intact.
  - **Markdown**: Heading-aware chunking preserving `#`, `##`, `###` hierarchy and code fences.
  - **Text / PDF**: Token-aware sliding window preserving exact start/end line numbers.
- **⚡ Zero-Infra Hybrid Storage**: Pure SQLite database (`.localrag/index.db`) combining Dense Vector Search with SQLite FTS5 (BM25) full-text search.
- **🎯 Reciprocal Rank Fusion (RRF)**: Merges semantic vector similarity with exact keyword lookup for 10x better precision on code symbols and error codes.

---

## 📦 Quickstart

### 1. Installation
```bash
git clone https://github.com/RitualDev-Lab/localrag-kit.git
cd localrag-kit
python -m venv .venv
# On Windows:
.\venv\Scripts\pip install -r requirements.txt
# On Linux/macOS:
source .venv/bin/activate && pip install -r requirements.txt
```

### 2. Scan & Preview Any Folder
```bash
python -m localrag.cli scan /path/to/project --preview
```

---

## 🧪 Running Tests
```bash
python -m pytest -v
```

---

## 📜 License
MIT © [RitualDev-Lab](https://github.com/RitualDev-Lab)
