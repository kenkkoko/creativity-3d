"""統整層：把 guardrails、embedding、計分、LLM、儲存串成完整流程。

analyze()          作答 → AnalysisResponse
create_stimulus()  新增自訂刺激詞（種子在背景產生）
seed_stimulus()    產生 AI 種子點並擬合投影（冷啟動）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from . import guardrails, llm_tasks, scoring
from .config import Settings
from .projection import Projector
from .providers.base import EmbeddingProvider, LLMProvider, ProviderError
from .schemas import (
    VERDICT_WEIGHT,
    AnalysisResponse,
    ErrorCode,
    LlmJudgement,
    RuleFlag,
    ScoredAssociation,
    Stimulus,
    SubmitRequest,
    Verdict,
)
from .store import Store

log = logging.getLogger(__name__)

RULE_MESSAGES = {
    RuleFlag.empty: "空白",
    RuleFlag.no_cjk: "需要包含中文",
    RuleFlag.gibberish: "看起來像亂打的字",
    RuleFlag.duplicate: "和前面的答案重複",
    RuleFlag.same_as_stimulus: "和題目相同",
    RuleFlag.blocked_word: "包含不適當的字詞",
    RuleFlag.injection: "包含指令性文字",
    RuleFlag.too_long: "超過 20 字",
}

FALLBACK_INTERPRETATION_NOTE = "解讀暫時無法產生，以下只顯示計算出的數字。"


class AppError(Exception):
    def __init__(self, code: ErrorCode, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


@dataclass
class Services:
    settings: Settings
    llm: LLMProvider
    embedder: EmbeddingProvider
    store: Store


# ---------------------------------------------------------------------------
# 作答分析
# ---------------------------------------------------------------------------


def analyze(svc: Services, req: SubmitRequest) -> AnalysisResponse:
    s, store = svc.settings, svc.store
    stim = store.get_stimulus(req.stimulus_id, req.client_session_id)
    if stim is None:
        raise AppError(ErrorCode.unknown_stimulus, "找不到這個刺激詞。", 404)
    if not stim.ready:
        raise AppError(ErrorCode.invalid_input, "這個詞的 AI 參考點還在準備中，請稍候再試。", 409)

    # 1) 遮蔽個資 → 規則層
    texts = [guardrails.mask_pii(a.text) for a in req.associations]
    notes = [guardrails.mask_pii(a.note) if a.note else None for a in req.associations]
    flags: list[RuleFlag] = []
    for i, t in enumerate(texts):
        flags.append(guardrails.check_association(t, stim.word, texts[:i]))
    if guardrails.batch_is_attack(flags):
        raise AppError(ErrorCode.blocked_input, "這次的作答包含太多指令性文字，無法分析。")

    # 2) 向量層
    try:
        mat = svc.embedder.embed([stim.word, *texts])
    except ProviderError as e:
        raise AppError(ErrorCode.llm_unavailable, "語義模型暫時無法使用，請確認 Ollama 已啟動。", 503) from e
    stim_vec, vecs = mat[0], mat[1:]
    dists = scoring.semantic_distances(stim_vec, vecs)
    outliers = scoring.outlier_flags(dists, store.pool_distances(stim.id))

    # 3) LLM 層（只送規則層通過的）
    warnings: list[str] = []
    ok_idx = [i for i, f in enumerate(flags) if f == RuleFlag.ok]
    judgements: dict[int, LlmJudgement] | None = {}
    if ok_idx:
        items = [
            llm_tasks.JudgeItem(index=i, text=texts[i], note=notes[i],
                                outlier=bool(outliers[i]), distance=float(dists[i]))
            for i in ok_idx
        ]
        judgements = llm_tasks.judge_associations(svc.llm, stim.word, items, max_retries=s.llm_max_retries)
        if judgements is None:
            warnings.append("合理性判斷暫時無法使用，已改用保守規則：距離極端的答案先算一半。")

    partial_fallback = False
    verdicts: list[Verdict] = []
    bridges: list[list[str]] = []
    reasons: list[str] = []
    for i, f in enumerate(flags):
        if f != RuleFlag.ok:
            verdicts.append(Verdict.unreasonable)
            bridges.append([])
            reasons.append(RULE_MESSAGES.get(f, "未通過檢查"))
        elif judgements is None:
            verdicts.append(Verdict.borderline if outliers[i] else Verdict.reasonable)
            bridges.append([])
            reasons.append("保守判斷（模型暫時無法使用）")
        elif i in judgements:
            j = judgements[i]
            verdicts.append(j.verdict)
            bridges.append(list(j.bridge))
            reasons.append(j.reason)
        else:  # 這一批判斷失敗 → 保守規則
            verdicts.append(Verdict.borderline if outliers[i] else Verdict.reasonable)
            bridges.append([])
            reasons.append("保守判斷（這一批模型判斷失敗）")
            partial_fallback = True
    if partial_fallback:
        warnings.append("部分答案的合理性判斷失敗，已改用保守規則。")
    weights = [VERDICT_WEIGHT[v] for v in verdicts]
    eff = [float(d) * w for d, w in zip(dists, weights)]

    # 4) 稀有度、投影
    valid = [i for i, w in enumerate(weights) if w > 0]
    rar_valid = scoring.rarity_per_item(
        vecs[valid] if valid else np.zeros((0, vecs.shape[1]), dtype=np.float32),
        store.pool_vectors_by_user(stim.id), s.similar_threshold, s.rarity_min_users,
    )
    rarity: list[float | None] = [None] * len(texts)
    for k, i in enumerate(valid):
        rarity[i] = rar_valid[k]

    projector = ensure_projector(svc, stim, stim_vec) or Projector.fit(stim_vec, vecs)

    # 聯想橋中間概念：一次 embedding，算真實座標
    terms = sorted({t for b in bridges for t in b})
    term_pos: dict[str, object] = {}
    if terms:
        try:
            tv = svc.embedder.embed(terms)
            td = scoring.semantic_distances(stim_vec, tv)
            term_pos = {t: projector.project(tv[k], float(td[k])) for k, t in enumerate(terms)}
        except ProviderError:
            log.warning("聯想橋座標計算失敗，前端改用插值")

    scored = [
        ScoredAssociation(
            index=i, text=texts[i], note=notes[i], rule_flag=flags[i], verdict=verdicts[i],
            weight=weights[i], distance=float(dists[i]), effective_distance=eff[i],
            rarity=rarity[i], outlier=bool(outliers[i]), bridge=bridges[i],
            bridge_positions=[term_pos[t] for t in bridges[i]] if all(t in term_pos for t in bridges[i]) else [],
            reason=reasons[i],
            position=projector.project(vecs[i], float(dists[i])),
        )
        for i in range(len(texts))
    ]

    # 5) 指紋與解讀
    fingerprint = scoring.build_fingerprint(
        effective_distances=eff, weights=weights,
        valid_vecs=vecs[valid] if valid else np.zeros((0, vecs.shape[1]), dtype=np.float32),
        rarities=rarity,
        population=store.pool_raw_fingerprints(stim.id, exclude_session=req.client_session_id),
        min_users_for_percentile=s.percentile_min_users,
    )
    interpretation = llm_tasks.interpret(svc.llm, stim.word, fingerprint, max_retries=s.llm_max_retries)
    if interpretation is None:
        warnings.append(FALLBACK_INTERPRETATION_NOTE)

    bad = sum(1 for w in weights if w == 0)
    if bad:
        warnings.append(f"有 {bad} 則答案沒有通過合理性檢查，不列入計分。")

    response = AnalysisResponse(
        submission_id="pending",
        stimulus=stim,
        associations=scored,
        fingerprint=fingerprint,
        interpretation=interpretation,
        pool=store.pool_stats(stim.id, s.percentile_min_users),
        cloud=store.cloud_points(stim.id),
        warnings=warnings,
        created_at=datetime.now(timezone.utc),
    )
    sid = store.save_submission(stimulus_id=stim.id, session_id=req.client_session_id,
                                response=response, vecs=vecs)
    return response.model_copy(update={"submission_id": sid})


# ---------------------------------------------------------------------------
# 刺激詞與種子
# ---------------------------------------------------------------------------


def seed_stimulus(svc: Services, stim: Stimulus) -> None:
    """產生 AI 種子點、擬合投影、標記可作答。失敗時仍標記可作答（以作答者自身向量擬合投影）。"""
    try:
        seeds = llm_tasks.generate_seeds(svc.llm, stim.word, n=30, max_retries=svc.settings.llm_max_retries)
        if seeds:
            mat = svc.embedder.embed([stim.word, *seeds])
            stim_vec, vecs = mat[0], mat[1:]
            d = scoring.semantic_distances(stim_vec, vecs)
            svc.store.add_seed_points(stim.id, seeds, vecs, distances=d)
            svc.store.save_projector(stim.id, Projector.fit(stim_vec, vecs).to_dict())
            log.info("種子完成：%s（%d 點）", stim.word, len(seeds))
        else:
            log.warning("種子產生失敗：%s", stim.word)
    except Exception:  # noqa: BLE001
        log.exception("種子流程錯誤：%s", stim.word)
    finally:
        svc.store.mark_ready(stim.id)


def ensure_projector(svc: Services, stim: Stimulus, stim_vec: np.ndarray | None = None) -> Projector | None:
    """取得最新版投影；舊版（v1，未置中）就用種子向量重新擬合並存回。沒有種子時回傳 None。"""
    data = svc.store.load_projector(stim.id)
    if Projector.is_current(data):
        return Projector.from_dict(data)
    seeds = svc.store.seed_vectors(stim.id)
    if seeds is None:
        return None
    if stim_vec is None:
        stim_vec = svc.embedder.embed([stim.word])[0]
    proj = Projector.fit(stim_vec, seeds)
    svc.store.save_projector(stim.id, proj.to_dict())
    log.info("投影已更新為 v%d：%s", Projector.VERSION, stim.word)
    return proj


def seed_missing_presets(svc: Services) -> None:
    for stim in svc.store.list_stimuli(None):
        if stim.source.value != "preset":
            continue
        if svc.store.seed_vectors(stim.id) is None:
            seed_stimulus(svc, stim)
        else:
            try:
                ensure_projector(svc, stim)
            except ProviderError:
                log.warning("無法更新投影：%s", stim.word)


def create_stimulus(svc: Services, word: str, session_id: str) -> tuple[Stimulus, bool]:
    """回傳 (刺激詞, 是否需要在背景產生種子)。"""
    word = word.strip()
    flag = guardrails.check_stimulus_word(word)
    if flag != RuleFlag.ok:
        raise AppError(ErrorCode.blocked_input, f"這個詞無法使用：{RULE_MESSAGES.get(flag, '未通過檢查')}。")

    existing = svc.store.find_stimulus_by_word(word)
    if existing is not None:
        visible = svc.store.get_stimulus(existing.id, session_id)
        if visible is not None:
            return visible, False

    if svc.store.count_created_today(session_id) >= svc.settings.max_custom_stimuli_per_day:
        raise AppError(ErrorCode.invalid_input,
                       f"每天最多新增 {svc.settings.max_custom_stimuli_per_day} 個詞，明天再來吧。", 429)

    check = llm_tasks.check_stimulus(svc.llm, word, max_retries=svc.settings.llm_max_retries)
    if check is None:
        raise AppError(ErrorCode.llm_unavailable, "目前無法檢查這個詞，請稍後再試。", 503)
    if not check.is_associable:
        raise AppError(ErrorCode.blocked_input, f"這個詞不適合當題目：{check.reason}")

    return svc.store.create_stimulus(word, session_id), True
