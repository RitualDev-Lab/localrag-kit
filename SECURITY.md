# Security Policy

The **localrag-kit** maintainer team considers privacy, air-gapped security, and document isolation fundamental design constraints. Because localrag-kit indexes sensitive local codebases, internal documentation, and proprietary PDFs, we enforce strict local-first security boundaries.

## Supported Versions

Security updates are actively maintained for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Security Vulnerability

If you identify a security vulnerability (such as unauthorized outbound network telemetry, SQLite/FTS5 injection vectors, local file traversal during ingestion, or model prompt injection risks):

1. **Do NOT report via public GitHub issues.** Public issues expose user environments to risk before a defensive patch can be distributed.
2. **Use GitHub Private Vulnerability Reporting**:
   - Navigate to the **[Security tab](https://github.com/RitualDev-Lab/localrag-kit/security)** on GitHub.
   - Click **"Report a vulnerability"** to submit a private security advisory.
3. **Alternative Direct Email**:
   - Email our security maintainers directly at:
     **`security@ritualdev.com`**
   - Please include:
     - Clear description of the vulnerability and affected subsystem (`localrag.storage`, `localrag.indexer`, `localrag.server`).
     - Reproduction steps or proof-of-concept (PoC).
     - Operating system, Python version, and execution environment.

## Response SLA

- **Initial Triage**: Within 24-48 hours.
- **Risk Assessment**: Within 72 hours.
- **Coordinated Disclosure**: Security patches will be published to PyPI immediately upon fix validation.

---

## ??? Core Security Invariants

- **100% Air-Gapped & Offline**: LocalRAG-Kit operates entirely on local compute. Core indexing and hybrid search make zero background network requests, zero telemetry calls, and zero external vector DB syncs.
- **Parameterization Against SQL Injection**: All queries to SQLite and the FTS5 full-text search virtual table strictly use parameterized queries (`?`). User query strings are never concatenated directly into raw SQL.
- **Safe Path Traversal Protection**: Ingestion paths are strictly resolved against verified local directories, preventing symlink traversal attacks outside designated repository boundaries.
- **Model Sandbox**: When running local models via Ollama or local GGUF runtimes, prompts and context chunks are passed through structured templates without shell evaluation.
