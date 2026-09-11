"""Mock LLM provider for unit testing, offline development, and deterministic synthesis."""

from typing import Iterator, Optional
from localrag.providers.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Deterministic LLM for testing without requiring local GPU or Ollama."""

    def __init__(self, model_name: str = "mock-agent"):
        self._name = model_name

    @property
    def model_name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        return "".join(self.stream_generate(prompt, system_prompt, temperature, max_tokens))

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        response = (
            f"Based on the provided context, here is the answer:\n\n"
            f"Summary: Verified and processed request.\n"
            f"Target Details: Extracted from local index chunks."
        )
        # Yield token by token for streaming simulation
        words = response.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
