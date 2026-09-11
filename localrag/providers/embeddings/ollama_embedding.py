"""Ollama local embedding provider connecting to local Ollama daemon."""

from typing import List, Optional
import httpx
from localrag.providers.base import BaseEmbeddingProvider
from localrag.storage.vector_ops import normalize_vector


DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
KNOWN_DIMENSIONS = {
    "nomic-embed-text": 768,
    "bge-m3": 1024,
    "all-minilm": 384,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
}


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Generates embeddings via local Ollama API server."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_EMBED_MODEL,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dim = KNOWN_DIMENSIONS.get(model, 768)

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    @property
    def dimension(self) -> int:
        return self._dim

    def is_available(self) -> bool:
        """Check if Ollama server is running and the target embedding model is pulled."""
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    models = res.json().get("models", [])
                    target = self.model.lower()
                    return any(
                        target in m.get("name", "").lower() or target in m.get("model", "").lower()
                        for m in models
                    )
                return False
        except Exception:
            return False

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text."""
        batch_res = self.embed_batch([text])
        return batch_res[0] if batch_res else [0.0] * self._dim

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding request using Ollama's /api/embed endpoint (or fallback to /api/embeddings)."""
        if not texts:
            return []

        # Try /api/embed (Ollama >= 0.1.30 supporting multi-input batching)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": texts},
                )
                if res.status_code == 200:
                    data = res.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings and len(embeddings) == len(texts):
                        self._dim = len(embeddings[0])
                        return [normalize_vector(emb) for emb in embeddings]
        except Exception:
            pass

        # Fallback to single-item /api/embeddings loop
        results: List[List[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for text in texts:
                try:
                    res = client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                    )
                    if res.status_code == 200:
                        emb = res.json().get("embedding", [])
                        if emb:
                            self._dim = len(emb)
                            results.append(normalize_vector(emb))
                            continue
                    results.append([0.0] * self._dim)
                except Exception:
                    results.append([0.0] * self._dim)

        return results
