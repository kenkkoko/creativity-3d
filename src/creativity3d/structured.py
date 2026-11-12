"""Structured Output 呼叫：Pydantic 驗證＋自動重試＋紀錄。

流程：
1. 把 Pydantic model 轉成 JSON Schema，交給模型（format=schema）
2. 用 Pydantic 驗證回傳內容（不做字串解析）
3. 失敗就把錯誤訊息附在下一次的提示中重試，最多 max_retries 次
4. 全部失敗回傳 None，由呼叫端改用預設結果
5. 每一次嘗試都寫進 logs/llm.jsonl
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .logging_setup import log_llm_call
from .providers.base import LLMProvider, ProviderError

T = TypeVar("T", bound=BaseModel)


def call_structured(
    llm: LLMProvider,
    model_cls: type[T],
    system: str,
    user: str,
    *,
    task: str,
    max_retries: int = 3,
    extra_check: Callable[[T], str | None] | None = None,
) -> T | None:
    """呼叫 LLM 並回傳通過驗證的 model；全部失敗回傳 None。

    extra_check：額外的語意檢查（例如比對 cited_numbers），
    回傳錯誤訊息字串代表不通過，回傳 None 代表通過。
    """
    schema = model_cls.model_json_schema()
    feedback = ""
    for attempt in range(1, max_retries + 1):
        prompt = user if not feedback else f"{user}\n\n上一次的輸出不合格：{feedback}\n請修正後重新輸出。"
        t0 = time.perf_counter()
        raw = ""
        error: str | None = None
        result: T | None = None
        try:
            raw = llm.chat_json(system, prompt, schema)
            result = model_cls.model_validate_json(raw)
            if extra_check is not None:
                error = extra_check(result)
                if error:
                    result = None
        except ValidationError as e:
            error = f"格式驗證失敗：{e.errors()[:3]}"
        except ProviderError as e:
            error = str(e)
        except Exception as e:  # noqa: BLE001 — 任何例外都要記錄並重試
            error = f"{type(e).__name__}: {e}"

        log_llm_call(
            task=task,
            provider=llm.name,
            attempt=attempt,
            max_retries=max_retries,
            ok=result is not None,
            error=error,
            elapsed_ms=round((time.perf_counter() - t0) * 1000),
            input={"system": system, "user": prompt},
            output=raw,
        )
        if result is not None:
            return result
        feedback = error or "未知錯誤"
    return None
