"""Abstract base classes for Embedding and LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator


class BaseEmbeddingProvider(ABC):
    """Interface for generating dense vector embeddings from text."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier name."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output dimensionality of the embedding vector."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate a dense normalized float vector for a single string."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
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

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a complete text response synchronously."""

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream generated text chunks in real-time."""

    def is_available(self) -> bool:
        """Check if provider backend is reachable."""
        return True
