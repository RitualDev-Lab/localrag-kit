"""Ollama local LLM provider supporting streaming and synchronous inference."""

import json
from typing import Iterator, Optional
import httpx
from localrag.providers.base import BaseLLMProvider


DEFAULT_OLLAMA_LLM_MODEL = "llama3.2"


class OllamaLLMProvider(BaseLLMProvider):
    """Executes local inference with Ollama daemon."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_LLM_MODEL,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return f"ollama/{self.model}"

    def is_available(self) -> bool:
        """Check if Ollama server is running and the target model is available."""
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

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Synchronously generate full response text."""
        chunks = list(self.stream_generate(prompt, system_prompt, temperature, max_tokens))
        return "".join(chunks)

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream generated tokens in real-time using Ollama JSON stream."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status_code != 200:
                        yield f"[Ollama Error: Status {response.status_code}]"
                        return

                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk_data = json.loads(line)
                            token = chunk_data.get("response", "")
                            if token:
                                yield token
                            if chunk_data.get("done", False):
                                break
                        except Exception:
                            continue
        except Exception as e:
            yield f"[Ollama Connection Error: {e}]"
