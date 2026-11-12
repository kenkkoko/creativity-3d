"""模型提供者的抽象介面。

換模型（Ollama → 雲端 API）時，只要新增一個實作這兩個介面的類別，
並在 .env 的 PROVIDER 指定它，其他程式都不用改。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def chat_json(self, system: str, user: str, json_schema: dict[str, Any]) -> str:
        """送出對話，要求模型依 JSON Schema 回傳（Structured Output）。

        回傳模型輸出的原始字串；驗證與重試由 structured.py 負責。
        """

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """回報模型是否可用。"""


class EmbeddingProvider(ABC):
    name: str = "base"

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """回傳 shape = (len(texts), dim) 的 L2 正規化向量。"""


class ProviderError(RuntimeError):
    """模型服務無法連線或回傳錯誤。"""
