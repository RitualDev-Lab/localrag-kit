"""OpenAI-compatible cloud embedding provider (OpenAI, Gemini, Together, Groq)."""

import os
from typing import List, Optional
import httpx
from localrag.providers.base import BaseEmbeddingProvider
from localrag.storage.vector_ops import normalize_vector


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Generates embeddings using OpenAI-compatible /v1/embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dim = 1536 if "3-small" in model or "ada-002" in model else 768

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    @property
    def dimension(self) -> int:
        return self._dim

    def is_available(self) -> bool:
        return bool(self.api_key)

    def embed_text(self, text: str) -> List[float]:
        batch = self.embed_batch([text])
        return batch[0] if batch else [0.0] * self._dim

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts or not self.api_key:
            return [[0.0] * self._dim for _ in texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": texts},
                )
                if res.status_code == 200:
                    data = res.json().get("data", [])
                    data.sort(key=lambda x: x.get("index", 0))
                    embeddings = [item.get("embedding", []) for item in data]
                    if embeddings:
                        self._dim = len(embeddings[0])
                        return [normalize_vector(emb) for emb in embeddings]
        except Exception:
            pass

        return [[0.0] * self._dim for _ in texts]
