"""Embedding and LLM provider layer for LocalRAG-Kit."""

from localrag.providers.base import BaseEmbeddingProvider, BaseLLMProvider
from localrag.providers.embeddings.local_hasher import FastFeatureEmbeddingProvider
from localrag.providers.embeddings.ollama_embedding import OllamaEmbeddingProvider
from localrag.providers.embeddings.openai_embedding import OpenAIEmbeddingProvider
from localrag.providers.llm.mock_llm import MockLLMProvider
from localrag.providers.llm.ollama_llm import OllamaLLMProvider
from localrag.providers.llm.openai_llm import OpenAILLMProvider
from localrag.providers.registry import get_embedding_provider, get_llm_provider

__all__ = [
    "BaseEmbeddingProvider",
    "BaseLLMProvider",
    "FastFeatureEmbeddingProvider",
    "MockLLMProvider",
    "OllamaEmbeddingProvider",
    "OllamaLLMProvider",
    "OpenAIEmbeddingProvider",
    "OpenAILLMProvider",
    "get_embedding_provider",
    "get_llm_provider",
]
