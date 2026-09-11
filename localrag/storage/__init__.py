"""Storage layer for LocalRAG-Kit."""

from localrag.storage.sqlite_store import SearchResult, SQLiteStore, StoreStats
from localrag.storage.vector_ops import (
    batch_cosine_similarities,
    cosine_similarity,
    deserialize_vector,
    normalize_vector,
    serialize_vector,
)

__all__ = [
    "SQLiteStore",
    "SearchResult",
    "StoreStats",
    "batch_cosine_similarities",
    "cosine_similarity",
    "deserialize_vector",
    "normalize_vector",
    "serialize_vector",
]
