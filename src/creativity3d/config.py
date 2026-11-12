"""設定：全部從 .env 讀取，不寫死在程式裡。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # 模型提供者：ollama（本機）或 fake（測試／離線用）
    provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    embed_model: str = "bge-m3"
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # 產品參數
    # 合理性門檻：False（預設）= AI 只負責畫聯想橋與標註，不影響計分，只有規則層（亂打、重複、指令）會擋；
    #            True = AI 判為不合理的答案不計分、勉強的算一半（v0.3 行為）
    appropriateness_gate: bool = False
    answer_seconds: int = 180
    percentile_min_users: int = 20
    rarity_min_users: int = 5
    similar_threshold: float = 0.85
    max_custom_stimuli_per_day: int = 5

    # 路徑
    data_dir: Path = PROJECT_ROOT / "data"
    log_dir: Path = PROJECT_ROOT / "logs"

    # 伺服器
    host: str = "127.0.0.1"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
