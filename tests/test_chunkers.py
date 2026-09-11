"""Tests for structure-aware, markdown, and code chunkers."""

from localrag.core.chunkers.code import CodeChunker
from localrag.core.chunkers.markdown import MarkdownChunker
from localrag.core.chunkers.sliding_window import SlidingWindowChunker
from localrag.core.chunkers import ChunkerRegistry
from localrag.core.models import Document, FileMetadata, FileType


def test_sliding_window_chunker_line_numbers():
    content = "\n".join([f"Line {i}: This is some sample content for chunking." for i in range(1, 50)])
    meta = FileMetadata(
        file_path="/dummy/test.txt",
        relative_path="test.txt",
        file_type=FileType.TEXT,
        sha256="123",
        line_count=49,
    )
    doc = Document(metadata=meta, content=content)

    chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata.start_line >= 1
        assert chunk.metadata.end_line >= chunk.metadata.start_line
        assert chunk.metadata.estimated_tokens > 0
        assert chunk.id.startswith("chunk_")


def test_markdown_chunker_headings_and_fences():
    md_content = """# Architecture Overview
LocalRAG-Kit is an offline RAG engine.
It stores data in SQLite.

## Ingestion Engine
The ingestion engine reads markdown and code.

```python
# Code fence inside ingestion
def hello():
    # Should not treat this as heading:
    # Heading inside code
    return True
```

## Storage Layer
Storage uses dense embeddings and BM25.
"""
    meta = FileMetadata(
        file_path="/dummy/README.md",
        relative_path="README.md",
        file_type=FileType.MARKDOWN,
        sha256="md123",
    )
    doc = Document(metadata=meta, content=md_content)

    chunker = MarkdownChunker(max_chunk_tokens=300)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3

    # Check section titles
    titles = [c.metadata.section_title for c in chunks]
    assert any("Architecture Overview" in (t or "") for t in titles)
    assert any("Ingestion Engine" in (t or "") for t in titles)
    assert any("Storage Layer" in (t or "") for t in titles)

    # Ensure code fence was preserved in ingestion chunk
    ingestion_chunk = next(c for c in chunks if "Ingestion Engine" in (c.metadata.section_title or ""))
    assert "def hello():" in ingestion_chunk.text
    assert "Heading inside code" in ingestion_chunk.text


def test_code_chunker_symbols():
    code_content = """import os
import sys

# Preamble imports above

class UserManager:
    def __init__(self):
        self.users = []

    def add_user(self, name):
        self.users.append(name)


def authenticate_user(token: str) -> bool:
    if not token:
        return False
    return True


async def fetch_profile(user_id: int):
    return {"id": user_id, "active": True}
"""
    meta = FileMetadata(
        file_path="/dummy/auth.py",
        relative_path="auth.py",
        file_type=FileType.CODE,
        sha256="py123",
        language="python",
    )
    doc = Document(metadata=meta, content=code_content)

    chunker = CodeChunker(max_chunk_tokens=200)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3

    symbols = [c.metadata.section_title for c in chunks]
    assert any("UserManager" in (s or "") for s in symbols)
    assert any("authenticate_user" in (s or "") for s in symbols)
    assert any("fetch_profile" in (s or "") for s in symbols)


def test_chunker_registry_dispatch():
    registry = ChunkerRegistry()

    doc_md = Document(
        metadata=FileMetadata(file_path="a.md", relative_path="a.md", file_type=FileType.MARKDOWN, sha256="1"),
        content="# Head\nText",
    )
    chunks_md = registry.chunk_document(doc_md)
    assert len(chunks_md) >= 1

    doc_py = Document(
        metadata=FileMetadata(file_path="a.py", relative_path="a.py", file_type=FileType.CODE, sha256="2"),
        content="def fn():\n    pass",
    )
    chunks_py = registry.chunk_document(doc_py)
    assert len(chunks_py) >= 1
