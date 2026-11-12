"""logging 設定。

- 一般訊息：輸出到終端機
- LLM 呼叫紀錄：每一次呼叫寫一行 JSON 到 logs/llm.jsonl
  （輸入、輸出、重試次數、耗時、是否成功），方便除錯與做 eval。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LLM_LOGGER_NAME = "creativity3d.llm"


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if not any(getattr(h, "_c3d", False) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        console._c3d = True  # type: ignore[attr-defined]
        root.addHandler(console)
        root.setLevel(level)

    llm_logger = logging.getLogger(LLM_LOGGER_NAME)
    if not any(getattr(h, "_c3d", False) for h in llm_logger.handlers):
        fh = logging.FileHandler(log_dir / "llm.jsonl", encoding="utf-8")
        fh.setFormatter(JsonLineFormatter())
        fh._c3d = True  # type: ignore[attr-defined]
        llm_logger.addHandler(fh)
        llm_logger.setLevel(logging.INFO)
        llm_logger.propagate = False


def log_llm_call(**data: Any) -> None:
    """記錄一次 LLM 呼叫。data 會原樣寫進 JSON。"""
    logging.getLogger(LLM_LOGGER_NAME).info("llm_call", extra={"data": data})
