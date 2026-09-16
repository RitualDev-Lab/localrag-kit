## ?? Summary of Changes

<!-- Provide a concise summary of the problem solved or feature added -->

---

## ?? Subsystems Affected

- [ ] `localrag.indexer` (Document harvesting, boundary chunking)
- [ ] `localrag.storage` (SQLite schemas, FTS5 BM25, vector BLOBs)
- [ ] `localrag.retrieval` (Dense cosine similarity, Reciprocal Rank Fusion)
- [ ] `localrag.providers` (Ollama, OpenAI, feature hasher embeddings)
- [ ] `localrag.server` / Web UI (FastAPI, SSE streaming, citation drawer)
- [ ] `localrag.cli` (Rich terminal interface, query commands)
- [ ] Documentation / Tests

---

## ?? Type of Change

- [ ] ?? **Bug Fix** (non-breaking fix)
- [ ] ? **New Feature** (new file parser, embedding provider, UI component)
- [ ] ? **Performance Improvement** (SIMD vector operations, batching)
- [ ] ??? **Security / Privacy Hardening**
- [ ] ?? **Tests Added / Updated**
- [ ] ?? **Documentation Update**

---

## ?? Verification & Air-Gapped Checklist

- [ ] **Tests Passing**: Ran `pytest tests/` locally and all tests passed.
- [ ] **Zero Telemetry**: Confirmed that no external tracking or outbound cloud requests are introduced.
- [ ] **SQL Parameterization**: All SQLite and FTS5 queries use parameterized inputs (`?`).
- [ ] **Code Formatting**: Passes formatting with `ruff check .` / `black --check .`.
- [ ] **Documentation**: Updated CLI usage and function docstrings where applicable.

---

## ?? Demonstration / Terminal Output (if applicable)

<!-- Attach screenshot of the Web UI or CLI output -->
