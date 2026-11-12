"""LLM 任務：合理性判斷、解讀、AI 種子、自訂詞檢查。

全部透過 `structured.call_structured`（Pydantic 驗證 + 重試 + 紀錄）。
每個任務另外有 extra_check 做語意檢查；不通過就把錯誤訊息回饋給模型重試。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import prompts
from .providers.base import LLMProvider
from .schemas import (
    Fingerprint,
    LlmInterpretation,
    LlmJudgement,
    LlmJudgementBatch,
    LlmSeedBatch,
    LlmStimulusCheck,
    RuleFlag,
    Verdict,
)
from .structured import call_structured

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_SEED_MIN, _SEED_MAX = 10, 40  # 對應 LlmSeedBatch 的 min_length / max_length


@dataclass
class JudgeItem:
    index: int
    text: str
    note: str | None
    outlier: bool
    distance: float


# ---------------------------------------------------------------------------
# 合理性判斷
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return "".join(s.split())


def _judge_check(stimulus: str, batch: list[JudgeItem]):
    expected = [it.index for it in batch]
    text_by_index = {it.index: it.text for it in batch}
    stim = _norm(stimulus)

    def check(result: LlmJudgementBatch) -> str | None:
        got = [j.index for j in result.judgements]
        if sorted(got) != sorted(expected):
            missing = sorted(set(expected) - set(got))
            extra = sorted(set(got) - set(expected))
            dup = sorted({i for i in got if got.count(i) > 1})
            return (
                f"index 必須剛好是 {expected}，每個一筆。"
                f"缺少 {missing}，多出 {extra}，重複 {dup}。"
            )
        for j in result.judgements:
            if j.verdict == Verdict.unreasonable and j.bridge:
                return f"index {j.index} 判為 unreasonable，bridge 必須是空陣列 []。"
            answer = _norm(text_by_index[j.index])
            for step in j.bridge:
                s = _norm(step)
                if not s:
                    return f"index {j.index} 的 bridge 有空字串。"
                if s == stim:
                    return f"index {j.index} 的 bridge 不可包含刺激詞本身「{stimulus}」。"
                if s == answer:
                    return f"index {j.index} 的 bridge 不可包含聯想本身「{text_by_index[j.index]}」。"
        return None

    return check


def judge_associations(
    llm: LLMProvider,
    stimulus: str,
    items: list[JudgeItem],
    max_retries: int = 3,
    batch_size: int = 15,
) -> dict[int, LlmJudgement] | None:
    """分批判斷；任何一批重試用盡仍失敗 → 回傳 None（統整者改用保守結果）。"""
    if not items:
        return {}
    batch_size = max(1, batch_size)
    out: dict[int, LlmJudgement] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        answers = [{"index": it.index, "text": it.text, "note": it.note} for it in batch]
        outliers = [it.index for it in batch if it.outlier]
        result = call_structured(
            llm,
            LlmJudgementBatch,
            prompts.JUDGE_SYSTEM,
            prompts.judge_user(stimulus, answers, outliers),
            task="judge",
            max_retries=max_retries,
            extra_check=_judge_check(stimulus, batch),
        )
        if result is None:
            return None
        for j in result.judgements:
            out[j.index] = j
    return out


# ---------------------------------------------------------------------------
# 解讀
# ---------------------------------------------------------------------------


def _fingerprint_numbers(fp: Fingerprint) -> set[float]:
    nums: set[float] = set()
    for name in Fingerprint.model_fields:
        dim = getattr(fp, name)
        for v in (dim.raw, dim.percentile):
            if v is not None:
                nums.add(round(float(v), 2))
    return nums


def _interpret_check(fp: Fingerprint):
    allowed = _fingerprint_numbers(fp)

    def check(result: LlmInterpretation) -> str | None:
        cited = {round(float(n), 2) for n in result.cited_numbers}
        bad = sorted(n for n in cited if n not in allowed)
        if bad:
            return f"cited_numbers 中的 {bad} 不在指紋資料裡，只能引用指紋中出現的數字。"
        texts = [result.headline, *result.strengths, *result.suggestions]
        uncited: list[str] = []
        for t in texts:
            for m in _NUMBER_RE.finditer(t):
                if round(float(m.group()), 2) not in cited:
                    uncited.append(m.group())
        if uncited:
            return (
                f"文字中出現的數字 {uncited} 沒有列進 cited_numbers"
                "（或不在指紋資料裡）。請只引用指紋中的數字，並全部列進 cited_numbers。"
            )
        return None

    return check


def interpret(
    llm: LLMProvider, stimulus: str, fingerprint: Fingerprint, max_retries: int = 3
) -> LlmInterpretation | None:
    return call_structured(
        llm,
        LlmInterpretation,
        prompts.INTERPRET_SYSTEM,
        prompts.interpret_user(stimulus, fingerprint.model_dump(mode="json")),
        task="interpret",
        max_retries=max_retries,
        extra_check=_interpret_check(fingerprint),
    )


# ---------------------------------------------------------------------------
# AI 種子
# ---------------------------------------------------------------------------


def generate_seeds(llm: LLMProvider, stimulus: str, n: int = 30, max_retries: int = 3) -> list[str]:
    """回傳去重、去空白、通過 guardrails.check_association 的種子；失敗回傳 []。"""
    from . import guardrails  # 延遲匯入：guardrails 由子任務 C 平行開發

    n = min(max(n, _SEED_MIN), _SEED_MAX)
    result = call_structured(
        llm,
        LlmSeedBatch,
        prompts.SEED_SYSTEM,
        prompts.seed_user(stimulus, n),
        task="seeds",
        max_retries=max_retries,
    )
    if result is None:
        return []
    kept: list[str] = []
    for raw in result.associations:
        text = raw.strip()
        if not text or text in kept:
            continue
        if guardrails.check_association(text, stimulus, kept) == RuleFlag.ok:
            kept.append(text)
    return kept


# ---------------------------------------------------------------------------
# 自訂刺激詞檢查
# ---------------------------------------------------------------------------


def check_stimulus(llm: LLMProvider, word: str, max_retries: int = 3) -> LlmStimulusCheck | None:
    return call_structured(
        llm,
        LlmStimulusCheck,
        prompts.STIMULUS_CHECK_SYSTEM,
        prompts.stimulus_check_user(word),
        task="stimulus_check",
        max_retries=max_retries,
    )
