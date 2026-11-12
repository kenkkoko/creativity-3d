"""依設定建立模型提供者。"""

from __future__ import annotations

from ..config import Settings
from .base import EmbeddingProvider, LLMProvider, ProviderError
from .fake import FakeEmbedding, FakeLLM
from .ollama import OllamaEmbedding, OllamaLLM

__all__ = ["EmbeddingProvider", "LLMProvider", "ProviderError", "build_providers"]


def build_providers(settings: Settings) -> tuple[LLMProvider, EmbeddingProvider]:
    if settings.provider == "fake":
        return FakeLLM(), FakeEmbedding()
    if settings.provider == "ollama":
        return (
            OllamaLLM(
                settings.ollama_base_url,
                settings.llm_model,
                settings.llm_temperature,
                settings.llm_timeout_seconds,
            ),
            OllamaEmbedding(
                settings.ollama_base_url, settings.embed_model, settings.llm_timeout_seconds
            ),
        )
    raise ValueError(f"不支援的 PROVIDER：{settings.provider}")
