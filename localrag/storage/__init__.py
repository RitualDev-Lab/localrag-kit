"""Storage layer for LocalRAG-Kit."""

from localrag.storage.sqlite_store import SQLiteStore, SearchResult, StoreStats
from localrag.storage.vector_ops import (
    cosine_similarity,
    batch_cosine_similarities,
    serialize_vector,
    deserialize_vector,
    normalize_vector,
)

__all__ = [
    "SQLiteStore",
    "SearchResult",
    "StoreStats",
    "cosine_similarity",
    "batch_cosine_similarities",
    "serialize_vector",
    "deserialize_vector",
    "normalize_vector",
]
