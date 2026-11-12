"""Ollama 實作：本機開源模型。

- LLM：POST /api/chat，format 帶入 JSON Schema（Structured Output）
- Embedding：POST /api/embed
"""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np

from .base import EmbeddingProvider, LLMProvider, ProviderError


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, temperature: float, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def chat_json(self, system: str, user: str, json_schema: dict[str, Any]) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": json_schema,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"Ollama 呼叫失敗：{e}") from e
        return r.json()["message"]["content"]

    def health(self) -> dict[str, Any]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
        except httpx.HTTPError as e:
            return {"ok": False, "error": f"連不到 Ollama（{self.base_url}）：{e}"}
        return {"ok": True, "installed_models": names}


class OllamaEmbedding(EmbeddingProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        try:
            r = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"Ollama embedding 失敗：{e}") from e
        return _normalize(np.asarray(r.json()["embeddings"], dtype=np.float32))
