"""Fast, 100% offline, standalone dense feature vector embedding provider."""

import hashlib
import math
import re
from typing import List, Sequence
from localrag.providers.base import BaseEmbeddingProvider
from localrag.storage.vector_ops import normalize_vector


class FastFeatureEmbeddingProvider(BaseEmbeddingProvider):
    """
    Zero-external-dependency local embedding generator using n-gram feature projections.
    Runs 100% offline, requires no downloads, and outputs normalized 384-dimensional vectors.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def name(self) -> str:
        return "fast-local"

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        """Generate a dense, normalized vector representation of input text."""
        if not text or not text.strip():
            return [0.0] * self._dim

        # Tokenize words and clean text
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
        if not tokens:
            return [0.0] * self._dim

        vec = [0.0] * self._dim

        # 1. Unigram feature hashing
        for token in tokens:
            self._project_token(token, weight=1.0, vec=vec)

        # 2. Bigram feature hashing for local word order context
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i + 1]}"
            self._project_token(bigram, weight=1.5, vec=vec)

        # 3. Subword character n-grams (3-grams) for typo tolerance and stemming
        for token in tokens:
            if len(token) >= 4:
                for j in range(len(token) - 2):
                    sub = token[j:j + 3]
                    self._project_token(sub, weight=0.4, vec=vec)

        return normalize_vector(vec)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    def _project_token(self, token: str, weight: float, vec: List[float]):
        """Hash token to feature index with random sign projection."""
        h = hashlib.sha256(token.encode("utf-8")).digest()
        # Derive two 32-bit integers from SHA-256
        idx1 = int.from_bytes(h[0:4], "little") % self._dim
        sign1 = 1.0 if (h[4] & 1) else -1.0
        vec[idx1] += sign1 * weight

        idx2 = int.from_bytes(h[8:12], "little") % self._dim
        sign2 = 1.0 if (h[12] & 1) else -1.0
        vec[idx2] += sign2 * (weight * 0.5)
