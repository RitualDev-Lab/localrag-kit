"""Vector serialization, normalization, and cosine similarity operations."""

import array
import math
from collections.abc import Sequence

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def serialize_vector(vector: Sequence[float]) -> bytes:
    """Convert a sequence of floats into a compact 32-bit float byte string."""
    if HAS_NUMPY:
        arr = np.asarray(vector, dtype=np.float32)
        return arr.tobytes()
    # Fallback to standard library array
    arr = array.array("f", vector)
    return arr.tobytes()


def deserialize_vector(blob: bytes) -> list[float]:
    """Convert a 32-bit float byte string back into a Python list of floats."""
    if not blob:
        return []
    if HAS_NUMPY:
        return np.frombuffer(blob, dtype=np.float32).tolist()
    arr = array.array("f")
    arr.frombytes(blob)
    return arr.tolist()


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Normalize a vector to unit length (L2 norm = 1.0)."""
    if not vector:
        return []
    if HAS_NUMPY:
        arr = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return vector if isinstance(vector, list) else list(vector)
        return (arr / norm).tolist()

    # Pure Python normalization
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return list(vector)
    return [x / norm for x in vector]


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    if HAS_NUMPY:
        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    dot = sum(x * y for x, y in zip(v1, v2))
    norm_a = math.sqrt(sum(x * x for x in v1))
    norm_b = math.sqrt(sum(y * y for y in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def batch_cosine_similarities(
    query_vector: Sequence[float],
    candidate_vectors: Sequence[Sequence[float]],
) -> list[float]:
    """Calculate cosine similarity against a batch of candidate vectors efficiently."""
    if not candidate_vectors:
        return []

    if HAS_NUMPY:
        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [0.0] * len(candidate_vectors)
        q_unit = q / q_norm

        matrix = np.asarray(candidate_vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        unit_matrix = matrix / norms
        scores = np.dot(unit_matrix, q_unit)
        return scores.tolist()

    # Fallback to single loop
    return [cosine_similarity(query_vector, c) for c in candidate_vectors]
