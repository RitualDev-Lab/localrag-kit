"""Abstract Base Class for text and document chunkers."""

from abc import ABC, abstractmethod
from typing import List
from localrag.core.models import Chunk, Document


class BaseChunker(ABC):
    """Interface for chunking an in-memory Document into discrete searchable Chunks."""

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document content into a list of Chunk objects with metadata."""
        pass

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count based on whitespace words and subword character heuristics."""
        if not text:
            return 0
        # English/code average is ~1.3 tokens per word or ~4 characters per token
        words = len(text.split())
        chars = len(text)
        return max(words, int(chars / 3.8))
