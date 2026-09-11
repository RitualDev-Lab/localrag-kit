"""Abstract base classes for Embedding and LLM providers."""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional


class BaseEmbeddingProvider(ABC):
    """Interface for generating dense vector embeddings from text."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier name."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output dimensionality of the embedding vector."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate a dense normalized float vector for a single string."""
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vectors for multiple strings. Default implementation loops embed_text."""
        return [self.embed_text(t) for t in texts]

    def is_available(self) -> bool:
        """Check if provider backend is reachable."""
        return True


class BaseLLMProvider(ABC):
    """Interface for text generation and streaming inference."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model being invoked."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a complete text response synchronously."""
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream generated text chunks in real-time."""
        pass

    def is_available(self) -> bool:
        """Check if provider backend is reachable."""
        return True
