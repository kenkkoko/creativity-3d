"""計分：把向量與合理性權重算成五維創造力指紋（SPEC 第 4 節）。

這裡全部是純函式，只依賴 numpy 與 schemas，不碰資料庫也不呼叫模型，
所以可以直接用小向量寫單元測試。

貫穿全檔的例子：刺激詞「蘋果」。
- 「水果」離蘋果很近（距離小）；「牛頓」要經過「掉落 → 萬有引力」才連得上（距離大）。
- 「化油器」被判定不合理，權重 0，所以有效距離 = 0，也不進入廣度、稀有度的計算。
"""

from __future__ import annotations

import numpy as np

from .schemas import DimensionScore, Fingerprint

_EPS = 1e-12
FINGERPRINT_KEYS = ("distance", "breadth", "rarity", "appropriateness", "fluency")


def _normalize_rows(vecs: np.ndarray) -> np.ndarray:
    """把每一列正規化成長度 1（零向量維持為零，避免除以 0）。"""
    vecs = np.asarray(vecs, dtype=np.float64)
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, _EPS)


def semantic_distances(stim_vec: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """計算每則聯想和刺激詞的語義距離 d = 1 − cos。

    向量照合約應已 L2 正規化；這裡仍會再正規化一次，避免上游漏做時算錯。
    結果裁切到 [0, 2]（cos 的範圍是 −1～1）。

    例子：刺激詞「蘋果」與聯想 [「水果」, 「牛頓」]，
    若 cos 分別是 0.9 和 0.4，回傳 array([0.1, 0.6])——牛頓跳得比較遠。

    回傳 shape (n,)；vecs 為空時回傳長度 0 的陣列。
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    if vecs.size == 0:
        return np.zeros(0, dtype=np.float64)
    stim = _normalize_rows(stim_vec)[0]
    cos = _normalize_rows(vecs) @ stim
    return np.clip(1.0 - cos, 0.0, 2.0)


def outlier_flags(
    distances: np.ndarray, pool_distances: np.ndarray, fallback: float = 0.9
) -> np.ndarray:
    """標記距離極端的答案（向量層，SPEC 第 5 節 ②）。

    - 公共分布（pool_distances）少於 20 筆：距離 > fallback（預設 0.9）就算極端。
    - 20 筆以上：距離 > Q3 + 1.5 × IQR 就算極端（盒鬚圖的上鬚）。

    例子：「蘋果」的公共分布 Q1 = 0.3、Q3 = 0.5，IQR = 0.2，門檻 = 0.5 + 0.3 = 0.8；
    「牛頓」距離 0.6 → False，「化油器」距離 0.85 → True（交給 LLM 時要求嚴格判斷）。

    回傳與 distances 等長的 bool 陣列。被標記不代表不合理，只是需要特別檢查。
    """
    distances = np.asarray(distances, dtype=np.float64)
    pool = np.asarray(pool_distances, dtype=np.float64).ravel()
    if pool.size < 20:
        threshold = fallback
    else:
        q1, q3 = np.percentile(pool, [25, 75])
        threshold = q3 + 1.5 * (q3 - q1)
    return distances > threshold


def breadth(valid_vecs: np.ndarray) -> float | None:
    """廣度：合理答案彼此之間 1 − cos 的平均（想得多散）。

    只傳入合理答案（w > 0）的向量；少於 3 則時回傳 None（兩點太少，看不出散不散）。

    例子：「蘋果」的答案 [「牛頓」, 「iPhone」, 「白雪公主」] 彼此都不相關，
    兩兩距離都約 0.8 → 廣度約 0.8；若答案是 [「紅蘋果」, 「青蘋果」, 「富士蘋果」]，
    彼此很像 → 廣度很低。
    """
    vecs = np.asarray(valid_vecs, dtype=np.float64)
    if vecs.ndim != 2 or vecs.shape[0] < 3:
        return None
    unit = _normalize_rows(vecs)
    sims = unit @ unit.T
    iu = np.triu_indices(unit.shape[0], k=1)
    return float(np.clip(1.0 - sims[iu], 0.0, 2.0).mean())


def rarity_per_item(
    vecs: np.ndarray,
    pool_by_user: dict[str, np.ndarray],
    threshold: float,
    min_users: int,
) -> list[float | None]:
    """每則答案的稀有度 r = 1 − (想到相似答案的人數 ÷ 總人數)。

    「相似」= 和該使用者任一答案的 cos ≥ threshold（預設 0.85）。一個人寫了很多相似答案
    也只算一次。pool_by_user 應只含其他已分享的人類使用者（不含 AI 種子、不含自己）。
    總人數 < min_users 時資料不足，全部回傳 None。

    例子：「蘋果」已有 10 人分享，其中 8 人寫過「水果」、1 人寫過「牛頓」，
    則「水果」的稀有度 = 1 − 8/10 = 0.2，「牛頓」= 1 − 1/10 = 0.9。
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    n = 0 if vecs.size == 0 else (1 if vecs.ndim == 1 else vecs.shape[0])
    total = len(pool_by_user)
    if total < min_users or total == 0:
        return [None] * n
    if n == 0:
        return []
    unit = _normalize_rows(vecs)
    hits = np.zeros(n, dtype=np.int64)
    for user_vecs in pool_by_user.values():
        user_vecs = np.asarray(user_vecs, dtype=np.float64)
        if user_vecs.size == 0:
            continue
        sims = unit @ _normalize_rows(user_vecs).T  # (n, m)
        hits += (sims >= threshold).any(axis=1)
    return [float(1.0 - h / total) for h in hits]


def percentile(value: float, population: list[float]) -> float:
    """百分位（0–100）：population 中小於 value 的比例，加上等於 value 的比例的一半，再 ×100。

    例子：value = 0.6，population = [0.2, 0.4, 0.6, 0.8]
    → 小於的有 2 個、等於的有 1 個 → (2 + 0.5) / 4 × 100 = 62.5。

    population 為空時沒有比較對象，丟出 ValueError（呼叫端應先檢查樣本數）。
    """
    if not population:
        raise ValueError("population 不可為空")
    arr = np.asarray(population, dtype=np.float64)
    less = float(np.sum(arr < value))
    equal = float(np.sum(arr == value))
    return float(np.clip((less + 0.5 * equal) / arr.size * 100.0, 0.0, 100.0))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def build_fingerprint(
    *,
    effective_distances: list[float],
    weights: list[float],
    valid_vecs: np.ndarray,
    rarities: list[float | None],
    population: list[dict[str, float | None]],
    min_users_for_percentile: int,
) -> Fingerprint:
    """把一次作答算成五維創造力指紋（不合成總分）。

    - distance（想得多遠）= effective_distances 的平均（不合理的答案以 0 計入）
    - appropriateness（合不合理）= weights 的平均
    - fluency（想了多少）= weight > 0 的則數
    - rarity（多少人也想到）= rarities 中非 None 值的平均；全為 None 時為 None
    - breadth（想得多散）= breadth(valid_vecs)
    - 百分位：population（同一刺激詞其他已分享使用者的原始值）人數 ≥ min_users_for_percentile
      才計算；某維度在 population 中為 None 的人略過，自己該維度為 None 時百分位也是 None。

    例子：「蘋果」答了 [牛頓 (d=0.6, w=1), 水果 (d=0.1, w=1), 化油器 (d=0.85, w=0)]
    → effective_distances = [0.6, 0.1, 0.0]，distance = 0.233；
      appropriateness = (1 + 1 + 0) / 3 = 0.667；fluency = 2。
    """
    raw: dict[str, float | None] = {
        "distance": _mean([float(x) for x in effective_distances]),
        "breadth": breadth(valid_vecs),
        "rarity": _mean([float(r) for r in rarities if r is not None]),
        "appropriateness": _mean([float(w) for w in weights]),
        "fluency": float(sum(1 for w in weights if w > 0)),
    }

    enough = len(population) >= min_users_for_percentile
    scores: dict[str, DimensionScore] = {}
    for key in FINGERPRINT_KEYS:
        value = raw[key]
        pct: float | None = None
        if enough and value is not None:
            others = [float(p[key]) for p in population if p.get(key) is not None]
            if others:
                pct = percentile(value, others)
        scores[key] = DimensionScore(raw=value, percentile=pct)
    return Fingerprint(**scores)
