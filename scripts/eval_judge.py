"""合理性判斷的黃金樣本評估（需要 Ollama）。

用法：
    uv run python scripts/eval_judge.py                     # 用 .env 裡的模型
    uv run python scripts/eval_judge.py --model qwen2.5:14b # 換模型比較

判定方式（和產品實際流程相同：先過規則層，再送 LLM）：
- expect=ok  ：verdict 不是 unreasonable 就算通過（reasonable / borderline 都可以）
- expect=bad ：verdict 必須是 unreasonable

輸出 eval_report.md：整體通過率、誤殺（合理卻被判不合理）、漏放（無關卻被放過）。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from creativity3d import guardrails, llm_tasks
from creativity3d.config import PROJECT_ROOT, get_settings
from creativity3d.logging_setup import setup_logging
from creativity3d.providers import build_providers
from creativity3d.schemas import RuleFlag, Verdict

CASES = PROJECT_ROOT / "tests" / "cases" / "golden_judge.jsonl"
REPORT = PROJECT_ROOT / "eval_report.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="覆寫 LLM_MODEL，例如 qwen2.5:14b")
    args = ap.parse_args()

    s = get_settings()
    if args.model:
        s = s.model_copy(update={"llm_model": args.model})
    setup_logging(s.log_dir)
    llm, embedder = build_providers(s)

    cases = [json.loads(l) for l in CASES.read_text("utf-8").splitlines() if l.strip()]
    by_stim: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_stim[c["stimulus"]].append(c)

    t0 = time.perf_counter()
    for stim, group in by_stim.items():
        texts = [c["text"] for c in group]
        dists = [0.5] * len(texts)
        try:
            mat = embedder.embed([stim, *texts])
            dists = [float(1 - (mat[0] @ v)) for v in mat[1:]]
        except Exception:  # noqa: BLE001 — embedding 不影響判斷結果，只用來標 outlier
            pass
        items = []
        for i, (c, d) in enumerate(zip(group, dists)):
            flag = guardrails.check_association(c["text"], stim, texts[:i])
            c["rule"] = flag.value
            if flag == RuleFlag.ok:
                items.append(llm_tasks.JudgeItem(index=i, text=c["text"], note=None, outlier=d > 0.9, distance=d))
        res = llm_tasks.judge_associations(llm, stim, items, max_retries=s.llm_max_retries) or {}
        for i, c in enumerate(group):
            if c["rule"] != RuleFlag.ok.value:
                c["verdict"], c["reason"] = Verdict.unreasonable.value, f"規則層：{c['rule']}"
            elif i in res:
                c["verdict"], c["reason"] = res[i].verdict.value, res[i].reason
            else:
                c["verdict"], c["reason"] = "error", "LLM 判斷失敗"
            c["pass"] = (c["verdict"] == "unreasonable") if c["expect"] == "bad" else (c["verdict"] in ("reasonable", "borderline"))
    elapsed = time.perf_counter() - t0

    ok_cases = [c for c in cases if c["expect"] == "ok"]
    bad_cases = [c for c in cases if c["expect"] == "bad"]
    rate = lambda xs: sum(c["pass"] for c in xs) / len(xs) * 100 if xs else 0.0  # noqa: E731
    lines = [
        "# 合理性判斷 eval 報告",
        "",
        f"- 模型：`{s.llm_model}`（embedding `{s.embed_model}`）",
        f"- 題數：{len(cases)}（應合理 {len(ok_cases)}、應不合理 {len(bad_cases)}）",
        f"- 耗時：{elapsed:.0f} 秒",
        "",
        "| 指標 | 通過率 |",
        "|---|---|",
        f"| 整體 | **{rate(cases):.1f}%** |",
        f"| 合理的答案沒有被誤殺 | {rate(ok_cases):.1f}% |",
        f"| 無關的答案有被擋下 | {rate(bad_cases):.1f}% |",
        "",
        "## 誤殺（合理卻被判不合理）",
        "",
        "| 刺激詞 | 聯想 | 判斷 | 理由 |",
        "|---|---|---|---|",
        *[f"| {c['stimulus']} | {c['text']} | {c['verdict']} | {c['reason']} |" for c in ok_cases if not c["pass"]],
        "",
        "## 漏放（無關卻被放過）",
        "",
        "| 刺激詞 | 聯想 | 判斷 | 理由 |",
        "|---|---|---|---|",
        *[f"| {c['stimulus']} | {c['text']} | {c['verdict']} | {c['reason']} |" for c in bad_cases if not c["pass"]],
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"\n完整報告 → {REPORT.name}")


if __name__ == "__main__":
    main()
