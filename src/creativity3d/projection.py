"""3D 投影：決定星星的「方向」，半徑則固定是真實語義距離（SPEC 4.1）。

做法：以刺激詞為原點，把「聯想向量 − 刺激詞向量」投影到 3 個主成分上，正規化成單位方向，
再乘上語義距離 d。這樣畫面上「離中心多遠」永遠等於真的語義距離，投影只影響方向。

例子：刺激詞「蘋果」，「牛頓」的距離 0.6。不管 PCA 把它轉到哪個方向，
它在星圖上與原點的距離都會是 0.6；「水果」距離 0.1，就一定比牛頓更靠近中心。
"""

from __future__ import annotations

import numpy as np

from .schemas import Point3D

_EPS = 1e-9
_SEED = 42
# 殘差投影為零向量（聯想和刺激詞幾乎一樣，或剛好與 3 個主成分都垂直）時使用的固定方向
FALLBACK_DIRECTION = np.array([0.0, 0.0, 1.0])


def _random_orthonormal(dim: int, k: int, existing: np.ndarray | None = None) -> np.ndarray:
    """用固定種子產生 k 個與 existing 互相正交的單位向量（回傳 shape (k, dim)）。"""
    rng = np.random.default_rng(_SEED)
    basis = [] if existing is None else list(existing)
    new: list[np.ndarray] = []
    while len(new) < k:
        v = rng.standard_normal(dim)
        for b in basis + new:  # Gram-Schmidt（做兩次以提高數值穩定度）
            v -= (v @ b) * b
        for b in basis + new:
            v -= (v @ b) * b
        n = np.linalg.norm(v)
        if n > _EPS:
            new.append(v / n)
    return np.array(new)


class Projector:
    """把高維 embedding 投影成 3D 座標。

    stim_vec：刺激詞的向量（原點）。components：shape (3, dim) 的正交單位基底。
    """

    VERSION = 2

    def __init__(self, stim_vec: np.ndarray, components: np.ndarray, center: np.ndarray | None = None):
        self.stim_vec = np.asarray(stim_vec, dtype=np.float64).ravel()
        self.components = np.asarray(components, dtype=np.float64)
        # center：參考答案殘差的平均。先減掉它再投影，方向才會在 3D 中散開，
        # 不會全部被「離開刺激詞的共同方向」主導而擠在同一側。
        self.center = (np.zeros_like(self.stim_vec) if center is None
                       else np.asarray(center, dtype=np.float64).ravel())
        if self.components.shape != (3, self.stim_vec.size):
            raise ValueError(
                f"components 形狀應為 (3, {self.stim_vec.size})，實際為 {self.components.shape}"
            )

    @classmethod
    def fit(cls, stim_vec: np.ndarray, reference_vecs: np.ndarray) -> "Projector":
        """以參考答案（AI 種子 + 已分享的公共答案）擬合 3 個主成分。

        對 (reference_vecs − stim_vec) 先置中（減去平均）再做 SVD。
        v1 沒有置中，結果所有點都擠在同一側（第一主成分 ≈ 共同的平均方向），v2 修正。
        取前 3 個右奇異向量。參考答案少於 3 筆，或有效秩不足 3 時，不足的軸用
        np.random.default_rng(42) 產生的正交向量補齊，所以結果可重現。
        每個主成分的正負號固定成「絕對值最大的分量為正」，避免 SVD 正負號任意造成星圖翻面。

        例子：「蘋果」有 30 個 AI 種子（水果、牛頓、iPhone…），第一主成分可能大致區分
        「食物」與「科技品牌」兩群，星圖上這兩群就會分在相對的兩側。
        """
        stim = np.asarray(stim_vec, dtype=np.float64).ravel()
        dim = stim.size
        if dim < 3:
            raise ValueError("向量維度至少要 3")
        ref = np.asarray(reference_vecs, dtype=np.float64)
        if ref.size == 0:
            ref = np.zeros((0, dim))
        ref = ref.reshape(-1, dim)

        comps = np.zeros((0, dim))
        center = np.zeros(dim)
        if ref.shape[0] >= 3:
            diffs = ref - stim
            center = diffs.mean(axis=0)
            diffs = diffs - center
            _, s, vt = np.linalg.svd(diffs, full_matrices=False)
            keep = s[:3] > _EPS * max(1.0, float(s[0]) if s.size else 1.0)
            comps = vt[:3][keep]
        if comps.shape[0] < 3:
            extra = _random_orthonormal(dim, 3 - comps.shape[0], comps if comps.size else None)
            comps = np.vstack([comps, extra]) if comps.size else extra

        # 固定正負號
        for i in range(3):
            j = int(np.argmax(np.abs(comps[i])))
            if comps[i, j] < 0:
                comps[i] = -comps[i]
        return cls(stim, comps, center)

    def project(self, vec: np.ndarray, distance: float) -> Point3D:
        """回傳 3D 座標：方向 = 主成分投影後正規化，與原點的距離 = distance。

        若投影結果是零向量（例如聯想向量和刺激詞完全相同），改用固定方向 (0, 0, 1)。

        例子：「蘋果 → 牛頓」distance = 0.6，回傳的點 (x, y, z) 滿足
        sqrt(x² + y² + z²) = 0.6。
        """
        diff = np.asarray(vec, dtype=np.float64).ravel() - self.stim_vec - self.center
        p = self.components @ diff
        n = float(np.linalg.norm(p))
        direction = p / n if n > _EPS else FALLBACK_DIRECTION
        r = max(float(distance), 0.0)
        x, y, z = (direction * r).tolist()
        return Point3D(x=x, y=y, z=z)

    def to_dict(self) -> dict:
        """轉成可存成 JSON 的 dict（存進 SQLite）。"""
        return {
            "version": self.VERSION,
            "stim_vec": self.stim_vec.tolist(),
            "components": self.components.tolist(),
            "center": self.center.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Projector":
        """從 to_dict() 的結果還原。"""
        return cls(np.asarray(d["stim_vec"]), np.asarray(d["components"]),
                   np.asarray(d["center"]) if "center" in d else None)

    @staticmethod
    def is_current(d: dict | None) -> bool:
        return bool(d) and d.get("version") == Projector.VERSION
