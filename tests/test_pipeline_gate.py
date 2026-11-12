"""合理性門檻開關：預設（關）時 AI 不擋答案，只有規則層會擋。"""

import json
import tempfile
from pathlib import Path

import pytest

from creativity3d.config import Settings
from creativity3d.pipeline import Services, analyze
from creativity3d.providers.base import LLMProvider
from creativity3d.providers.fake import FakeEmbedding
from creativity3d.schemas import Stimulus, StimulusSource, SubmitRequest
from creativity3d.store import Store


class JudgeAllUnreasonable(LLMProvider):
    """AI 把每一則都判為不合理（模擬模型過嚴）。"""
    name = "strict"

    def health(self):
        return {"ok": True}

    def chat_json(self, system, user, schema):
        if schema.get("title") == "LlmJudgementBatch":
            import re
            idx = [int(x) for x in re.findall(r"\d+", user.split("index 依序為")[1])]
            return json.dumps({"judgements": [{"index": i, "verdict": "unreasonable", "bridge": [], "reason": "看不出連結"} for i in idx]}, ensure_ascii=False)
        return json.dumps({"headline": "測試", "strengths": ["測試內容一"], "suggestions": ["測試內容二"], "cited_numbers": []}, ensure_ascii=False)


@pytest.mark.parametrize("gate, expected_counted", [(False, 2), (True, 0)])
def test_gate(gate, expected_counted, tmp_path):
    s = Settings(provider="fake", data_dir=tmp_path, log_dir=tmp_path / "logs", appropriateness_gate=gate)
    store = Store(tmp_path / "db.sqlite")
    store.ensure_presets([Stimulus(id="s07", word="紅色", source=StimulusSource.preset, abstraction_level=7)])
    svc = Services(s, JudgeAllUnreasonable(), FakeEmbedding(), store)
    req = SubmitRequest(stimulus_id="s07", elapsed_seconds=30, client_session_id="sessionGATE",
                        associations=[{"text": "櫻桃"}, {"text": "MSI"}, {"text": "哈哈哈"}])
    r = analyze(svc, req)
    weights = [a.weight for a in r.associations]
    assert sum(1 for w in weights if w > 0) == expected_counted
    assert r.associations[2].weight == 0  # 亂打永遠不計分
    assert r.fingerprint.fluency.raw == expected_counted
    assert r.fingerprint.appropriateness.raw == 0  # AI 看不出連結 → 合不合理維度為 0
