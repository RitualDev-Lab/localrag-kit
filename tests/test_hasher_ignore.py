"""Tests for hashing utilities and .gitignore/.ragignore filters."""

import tempfile
from pathlib import Path
from localrag.utils.hasher import compute_content_sha256, compute_file_sha256, generate_chunk_id
from localrag.utils.ignore import IgnoreFilter


def test_content_sha256():
    hash1 = compute_content_sha256("hello world")
    hash2 = compute_content_sha256("hello world")
    hash3 = compute_content_sha256("hello world 2")

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64


def test_file_sha256():
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(b"LocalRAG test content\nline 2")
        f.flush()
        temp_path = f.name

    try:
        file_hash = compute_file_sha256(temp_path)
        content_hash = compute_content_sha256("LocalRAG test content\nline 2")
        assert file_hash == content_hash
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_generate_chunk_id():
    id1 = generate_chunk_id("src/main.py", 1, 20, "def foo(): pass")
    id2 = generate_chunk_id("src/main.py", 1, 20, "def foo(): pass")
    id3 = generate_chunk_id("src/main.py", 21, 40, "def bar(): pass")

    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("chunk_")


def test_ignore_filter_defaults():
    with tempfile.TemporaryDirectory() as tmpdir:
        filt = IgnoreFilter(tmpdir)

        # Default ignored paths
        assert filt.is_ignored(".git/config") is True
        assert filt.is_ignored("node_modules/express/index.js") is True
        assert filt.is_ignored("__pycache__/test.cpython-312.pyc") is True
        assert filt.is_ignored(".venv/bin/python") is True
        assert filt.is_ignored(".localrag/index.db") is True
        assert filt.is_ignored("build/lib/module.py") is True
        assert filt.is_ignored("image.png") is True
        assert filt.is_ignored("archive.zip") is True

        # Valid allowed paths
        assert filt.is_ignored("README.md") is False
        assert filt.is_ignored("src/core/models.py") is False
        assert filt.is_ignored("docs/architecture.md") is False


def test_ignore_filter_custom_and_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / ".gitignore").write_text("secrets/\n*.tmp\n")
        (root / ".ragignore").write_text("legacy/**\n")

        filt = IgnoreFilter(root, custom_patterns=["custom_ignore/*"])

        assert filt.is_ignored("secrets/keys.json") is True
        assert filt.is_ignored("cache.tmp") is True
        assert filt.is_ignored("legacy/old_code.py") is True
        assert filt.is_ignored("custom_ignore/test.py") is True
        assert filt.is_ignored("src/active.py") is False
