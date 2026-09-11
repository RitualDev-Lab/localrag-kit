"""Fast cryptographic hashing utilities for files, content, and chunk IDs."""

import hashlib
from pathlib import Path


def compute_content_sha256(content: str | bytes) -> str:
    """Compute SHA-256 hash string for raw string or byte content."""
    hasher = hashlib.sha256()
    if isinstance(content, str):
        hasher.update(content.encode("utf-8", errors="ignore"))
    else:
        hasher.update(content)
    return hasher.hexdigest()


def compute_file_sha256(file_path: str | Path, block_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file on disk in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_chunk_id(file_rel_path: str, start_line: int, end_line: int, text: str) -> str:
    """Generate deterministic unique ID for a chunk based on origin and content."""
    seed = f"{file_rel_path}#{start_line}:{end_line}#{text[:64]}"
    hasher = hashlib.sha256(seed.encode("utf-8", errors="ignore"))
    return f"chunk_{hasher.hexdigest()[:16]}"
