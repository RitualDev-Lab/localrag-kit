"""Factory and provider registry for Embedding and LLM backends."""

from localrag.providers.base import BaseEmbeddingProvider, BaseLLMProvider
from localrag.providers.embeddings.local_hasher import FastFeatureEmbeddingProvider
from localrag.providers.embeddings.ollama_embedding import OllamaEmbeddingProvider
from localrag.providers.embeddings.openai_embedding import OpenAIEmbeddingProvider
from localrag.providers.llm.mock_llm import MockLLMProvider
from localrag.providers.llm.ollama_llm import OllamaLLMProvider
from localrag.providers.llm.openai_llm import OpenAILLMProvider


def get_embedding_provider(
    provider_type: str = "auto",
    model: str | None = None,
    **kwargs,
) -> BaseEmbeddingProvider:
    """
    Resolve embedding provider.
    Modes:
      - 'auto': Use Ollama if running, otherwise use fast local offline hasher.
      - 'fast' / 'local': Zero-dependency standalone 384-dim feature vectorizer.
      - 'ollama': Local Ollama embedding service (e.g. nomic-embed-text).
      - 'openai': OpenAI-compatible /v1/embeddings API.
    """
    p_type = provider_type.lower()

    if p_type == "auto":
        ollama = OllamaEmbeddingProvider(model=model or "nomic-embed-text", **kwargs)
        if ollama.is_available():
            return ollama
        return FastFeatureEmbeddingProvider(**kwargs)

    if p_type in ("fast", "local", "hash"):
        return FastFeatureEmbeddingProvider(**kwargs)

    if p_type == "ollama":
        m = model or "nomic-embed-text"
        return OllamaEmbeddingProvider(model=m, **kwargs)

    if p_type in ("openai", "cloud"):
        m = model or "text-embedding-3-small"
        return OpenAIEmbeddingProvider(model=m, **kwargs)

    raise ValueError(f"Unknown embedding provider type: {provider_type}")


def get_llm_provider(
    provider_type: str = "auto",
    model: str | None = None,
    **kwargs,
) -> BaseLLMProvider:
    """
    Resolve LLM inference provider.
    Modes:
      - 'auto': Use Ollama if running, otherwise use deterministic Mock LLM.
      - 'ollama': Local Ollama inference service (e.g. llama3.2, qwen2.5-coder).
      - 'openai': OpenAI-compatible /v1/chat/completions.
      - 'mock': Offline deterministic simulator for testing.
    """
    p_type = provider_type.lower()

    if p_type == "auto":
        ollama = OllamaLLMProvider(model=model or "llama3.2", **kwargs)
        if ollama.is_available():
            return ollama
        return MockLLMProvider(**kwargs)

    if p_type == "ollama":
        m = model or "llama3.2"
        return OllamaLLMProvider(model=m, **kwargs)

    if p_type in ("openai", "cloud"):
        m = model or "gpt-4o-mini"
        return OpenAILLMProvider(model=m, **kwargs)

    if p_type == "mock":
        return MockLLMProvider(**kwargs)

    raise ValueError(f"Unknown LLM provider type: {provider_type}")
