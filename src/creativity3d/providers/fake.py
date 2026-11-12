"""Fake 實作：不需要 Ollama，用於測試與離線開發。

- Embedding：以文字雜湊產生固定向量（同一個字永遠得到同一個向量）
- LLM：依 JSON Schema 產生最小的合法回應；可注入預先寫好的回應來模擬壞 JSON
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any

import numpy as np

from .base import EmbeddingProvider, LLMProvider


class FakeEmbedding(EmbeddingProvider):
    name = "fake"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, scripted: list[str] | None = None):
        # scripted：依序回傳的原始字串（用來測試壞 JSON、重試）
        self.scripted: deque[str] = deque(scripted or [])
        self.calls: list[dict[str, Any]] = []

    def chat_json(self, system: str, user: str, json_schema: dict[str, Any]) -> str:
        self.calls.append({"system": system, "user": user})
        if self.scripted:
            return self.scripted.popleft()
        return json.dumps(_minimal_instance(json_schema, json_schema), ensure_ascii=False)

    def health(self) -> dict[str, Any]:
        return {"ok": True, "installed_models": ["fake"]}


def _minimal_instance(schema: dict[str, Any], root: dict[str, Any]) -> Any:
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return _minimal_instance(root["$defs"][name], root)
    if "anyOf" in schema:
        return _minimal_instance(schema["anyOf"][0], root)
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object":
        return {
            k: _minimal_instance(v, root)
            for k, v in schema.get("properties", {}).items()
            if k in schema.get("required", [])
        }
    if t == "array":
        n = schema.get("minItems", 0)
        return [_minimal_instance(schema.get("items", {}), root) for _ in range(n)]
    if t == "string":
        return "測" * max(1, schema.get("minLength", 1))
    if t == "integer":
        return int(schema.get("minimum", 0))
    if t == "number":
        return float(schema.get("minimum", 0))
    if t == "boolean":
        return True
    return None
