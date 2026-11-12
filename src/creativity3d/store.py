"""SQLite 儲存：刺激詞、AI 種子、投影器、作答與公共分布。

只用標準庫 sqlite3。每個方法各自開一條連線，FastAPI 多執行緒下也安全。
- 向量以 numpy float32 的 tobytes() 存成 BLOB。
- 時間一律存 UTC 的 ISO 8601 字串。
- 文字一律是「已遮蔽個資」後的版本（由呼叫端先處理）。

「使用者」以匿名 session_id 區分：同一個 session 對同一刺激詞分享多次，仍只算 1 人。
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .projection import Projector
from .schemas import AnalysisResponse, CloudPoint, HistoryItem, PoolStats, Stimulus, StimulusSource

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stimuli (
    id TEXT PRIMARY KEY,
    word TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('preset', 'custom')),
    abstraction_level INTEGER,
    source_note TEXT NOT NULL DEFAULT '',
    ready INTEGER NOT NULL DEFAULT 0,
    published INTEGER NOT NULL DEFAULT 0,
    creator_session TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stimuli_word ON stimuli(word);
CREATE INDEX IF NOT EXISTS idx_stimuli_creator ON stimuli(creator_session, created_at);

CREATE TABLE IF NOT EXISTS players (
    nickname_key TEXT PRIMARY KEY,      -- 小寫後的暱稱，用來判斷重複（Kai = kai）
    nickname TEXT NOT NULL,
    pin_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    session_id TEXT NOT NULL UNIQUE,    -- 玩家的固定 ID（之後所有 API 用它）
    failed_count INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS seeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stimulus_id TEXT NOT NULL REFERENCES stimuli(id),
    text TEXT NOT NULL,
    vec BLOB NOT NULL,
    distance REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_seeds_stim ON seeds(stimulus_id);

CREATE TABLE IF NOT EXISTS projectors (
    stimulus_id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS submissions (
    id TEXT PRIMARY KEY,
    stimulus_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    shared INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    shared_at TEXT,
    fingerprint_raw TEXT NOT NULL,
    response_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sub_stim ON submissions(stimulus_id, shared);
CREATE INDEX IF NOT EXISTS idx_sub_session ON submissions(session_id, created_at);

CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id TEXT NOT NULL REFERENCES submissions(id),
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    weight REAL NOT NULL,
    distance REAL NOT NULL,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_answers_sub ON answers(submission_id);
"""

_FP_KEYS = ("distance", "breadth", "rarity", "appropriateness", "fluency")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).ravel().tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()



# ---------------------------------------------------------------------------
# 玩家身分（暱稱 + PIN）與個人紀錄
# ---------------------------------------------------------------------------

PIN_MAX_FAILURES = 5
PIN_LOCK_MINUTES = 10


def _clean_legacy_response(data: dict) -> dict:
    """舊紀錄相容：早期版本的解讀可能含空白或過短的項目（例如 "\\r\\n    "），
    讀取時去掉；清完若某個清單變空，就把整個解讀設為 None（前端改顯示預設文字）。"""
    interp = data.get("interpretation")
    if isinstance(interp, dict):
        for key in ("strengths", "suggestions"):
            items = [t.strip() for t in interp.get(key) or [] if isinstance(t, str) and len(t.strip()) >= 4]
            interp[key] = items
        if not interp.get("strengths") or not interp.get("suggestions"):
            data["interpretation"] = None
    return data


def _hash_pin(pin: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), 120_000).hex()


class LoginResult:
    def __init__(self, status: str, nickname: str = "", session_id: str = "", is_new: bool = False,
                 retry_minutes: int = 0):
        self.status = status  # ok / wrong_pin / locked
        self.nickname, self.session_id, self.is_new = nickname, session_id, is_new
        self.retry_minutes = retry_minutes


class _PlayerMixin:
    """玩家身分與個人紀錄。需要子類別提供 self._conn()。"""

    def login_player(self, nickname: str, pin: str, client_session_id: str) -> LoginResult:
        """暱稱不存在 → 建立（沿用目前瀏覽器的 session，舊紀錄自動延續）；
        已存在 → 驗證 PIN，成功後把目前瀏覽器的匿名紀錄併入玩家。
        連錯 5 次鎖 10 分鐘。

        例：Kai 在家裡電腦第一次輸入「Kai / 1234」→ 建立身分，家裡之前玩的紀錄都算 Kai 的；
        之後在公司電腦輸入「Kai / 1234」→ 拿回同一個 session_id，看得到全部紀錄。
        """
        key = nickname.casefold()
        now = datetime.now(timezone.utc)
        with self._conn() as con:
            row = con.execute("SELECT * FROM players WHERE nickname_key = ?", (key,)).fetchone()
            if row is None:
                # 目前的 session 若已屬於別的玩家，就換一個新的，避免兩人共用
                taken = con.execute("SELECT 1 FROM players WHERE session_id = ?", (client_session_id,)).fetchone()
                sid = uuid.uuid4().hex if taken else client_session_id
                salt = secrets.token_hex(16)
                con.execute(
                    "INSERT INTO players (nickname_key, nickname, pin_hash, salt, session_id, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (key, nickname, _hash_pin(pin, salt), salt, sid, _now()),
                )
                return LoginResult("ok", nickname, sid, True)

            if row["locked_until"] and datetime.fromisoformat(row["locked_until"]) > now:
                left = datetime.fromisoformat(row["locked_until"]) - now
                return LoginResult("locked", retry_minutes=max(1, int(left.total_seconds() // 60) + 1))

            if not secrets.compare_digest(_hash_pin(pin, row["salt"]), row["pin_hash"]):
                fails = row["failed_count"] + 1
                locked = (now + timedelta(minutes=PIN_LOCK_MINUTES)).isoformat() if fails >= PIN_MAX_FAILURES else None
                con.execute("UPDATE players SET failed_count = ?, locked_until = ? WHERE nickname_key = ?",
                            (0 if locked else fails, locked, key))
                if locked:
                    return LoginResult("locked", retry_minutes=PIN_LOCK_MINUTES)
                return LoginResult("wrong_pin")

            con.execute("UPDATE players SET failed_count = 0, locked_until = NULL WHERE nickname_key = ?", (key,))
            sid = row["session_id"]
            # 把這個瀏覽器的匿名紀錄併進來（但不搶別的玩家的紀錄）
            if client_session_id != sid:
                other = con.execute("SELECT 1 FROM players WHERE session_id = ?", (client_session_id,)).fetchone()
                if other is None:
                    con.execute("UPDATE submissions SET session_id = ? WHERE session_id = ?", (sid, client_session_id))
                    con.execute("UPDATE stimuli SET creator_session = ? WHERE creator_session = ?",
                                (sid, client_session_id))
            return LoginResult("ok", row["nickname"], sid, False)

    def player_nickname(self, session_id: str) -> str | None:
        with self._conn() as con:
            row = con.execute("SELECT nickname FROM players WHERE session_id = ?", (session_id,)).fetchone()
        return row["nickname"] if row else None

    def history(self, session_id: str, limit: int = 100) -> list[HistoryItem]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT s.id, s.stimulus_id, s.created_at, s.shared, s.fingerprint_raw, s.response_json, t.word"
                " FROM submissions s JOIN stimuli t ON t.id = s.stimulus_id"
                " WHERE s.session_id = ? ORDER BY s.created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        items = []
        for r in rows:
            fp = json.loads(r["fingerprint_raw"])
            interp = json.loads(r["response_json"]).get("interpretation") or {}
            items.append(HistoryItem(
                submission_id=r["id"], stimulus_id=r["stimulus_id"], stimulus_word=r["word"],
                created_at=datetime.fromisoformat(r["created_at"]),
                fluency=fp.get("fluency"), distance=fp.get("distance"),
                headline=interp.get("headline"), shared=bool(r["shared"]),
            ))
        return items

    def get_submission(self, submission_id: str, session_id: str) -> tuple[AnalysisResponse, bool] | None:
        """只有本人拿得到；回傳 (當時的分析結果, 是否已分享)。"""
        with self._conn() as con:
            row = con.execute(
                "SELECT response_json, shared FROM submissions WHERE id = ? AND session_id = ?",
                (submission_id, session_id),
            ).fetchone()
        if row is None:
            return None
        data = _clean_legacy_response(json.loads(row["response_json"]))
        return AnalysisResponse.model_validate(data), bool(row["shared"])

    def delete_submission(self, submission_id: str, session_id: str) -> bool:
        """刪除一筆作答（只有本人可以）。已分享的話，這筆答案也會一起從公共星雲移除。
        回傳是否真的刪到。"""
        with self._conn() as con:
            row = con.execute("SELECT 1 FROM submissions WHERE id = ? AND session_id = ?",
                              (submission_id, session_id)).fetchone()
            if row is None:
                return False
            con.execute("DELETE FROM answers WHERE submission_id = ?", (submission_id,))
            con.execute("DELETE FROM submissions WHERE id = ?", (submission_id,))
        return True

    def answer_vectors(self, submission_id: str) -> np.ndarray | None:
        """依 idx 順序回傳這次作答每則答案的向量。"""
        with self._conn() as con:
            rows = con.execute("SELECT vec FROM answers WHERE submission_id = ? ORDER BY idx",
                               (submission_id,)).fetchall()
        if not rows:
            return None
        return np.vstack([_from_blob(r["vec"]) for r in rows])


class Store(_PlayerMixin):
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            with con:  # 自動 commit / rollback
                yield con
        finally:
            con.close()

    # ------------------------------------------------------------------
    # 刺激詞
    # ------------------------------------------------------------------

    _STIM_SELECT = """
        SELECT s.*, (
            SELECT COUNT(DISTINCT sub.session_id) FROM submissions sub
            WHERE sub.stimulus_id = s.id AND sub.shared = 1
        ) AS response_count
        FROM stimuli s
    """

    @staticmethod
    def _row_to_stimulus(row: sqlite3.Row) -> Stimulus:
        if row["kind"] == "preset":
            source = StimulusSource.preset
        elif row["published"]:
            source = StimulusSource.community
        else:
            source = StimulusSource.mine
        return Stimulus(
            id=row["id"],
            word=row["word"],
            source=source,
            abstraction_level=row["abstraction_level"],
            source_note=row["source_note"] or "",
            response_count=int(row["response_count"] or 0),
            ready=bool(row["ready"]),
        )

    @staticmethod
    def _visible(row: sqlite3.Row, session_id: str | None) -> bool:
        if row["kind"] == "preset" or row["published"]:
            return True
        return session_id is not None and row["creator_session"] == session_id

    def ensure_presets(self, presets: list[Stimulus]) -> None:
        """寫入（或更新）預設詞庫；可重複呼叫。"""
        now = _now()
        with self._conn() as con:
            for p in presets:
                con.execute(
                    """
                    INSERT INTO stimuli (id, word, kind, abstraction_level, source_note,
                                         ready, published, creator_session, created_at)
                    VALUES (?, ?, 'preset', ?, ?, 1, 1, NULL, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        word = excluded.word,
                        abstraction_level = excluded.abstraction_level,
                        source_note = excluded.source_note
                    """,
                    (p.id, p.word, p.abstraction_level, p.source_note, now),
                )

    def list_stimuli(self, session_id: str | None) -> list[Stimulus]:
        """預設詞（依代號）+ 社群詞（依作答人數）+ 該 session 尚未公開的私人詞（依建立時間）。"""
        with self._conn() as con:
            rows = con.execute(self._STIM_SELECT).fetchall()
        rows = [r for r in rows if self._visible(r, session_id)]
        presets = sorted((r for r in rows if r["kind"] == "preset"), key=lambda r: r["id"])
        community = sorted(
            (r for r in rows if r["kind"] == "custom" and r["published"]),
            key=lambda r: (-int(r["response_count"] or 0), r["created_at"]),
        )
        mine = sorted(
            (r for r in rows if r["kind"] == "custom" and not r["published"]),
            key=lambda r: r["created_at"],
        )
        return [self._row_to_stimulus(r) for r in presets + community + mine]

    def get_stimulus(self, stimulus_id: str, session_id: str | None) -> Stimulus | None:
        """取得單一刺激詞；別人尚未公開的私人詞回傳 None。"""
        with self._conn() as con:
            row = con.execute(self._STIM_SELECT + " WHERE s.id = ?", (stimulus_id,)).fetchone()
        if row is None or not self._visible(row, session_id):
            return None
        return self._row_to_stimulus(row)

    def find_stimulus_by_word(self, word: str) -> Stimulus | None:
        """依文字找現有刺激詞（不分可見度，預設詞優先）；用於新增前檢查重複。

        注意：找到別人未公開的私人詞時也會回傳（source=mine），由呼叫端決定如何處理。
        """
        with self._conn() as con:
            row = con.execute(
                self._STIM_SELECT
                + " WHERE s.word = ? ORDER BY (s.kind = 'preset') DESC, s.published DESC,"
                " s.created_at LIMIT 1",
                (word.strip(),),
            ).fetchone()
        return self._row_to_stimulus(row) if row else None

    def create_stimulus(self, word: str, session_id: str) -> Stimulus:
        """建立自訂詞：id = "u" + 8 碼 hex，source=mine、ready=False、未分級。"""
        with self._conn() as con:
            while True:
                sid = "u" + uuid.uuid4().hex[:8]
                if con.execute("SELECT 1 FROM stimuli WHERE id = ?", (sid,)).fetchone() is None:
                    break
            con.execute(
                """
                INSERT INTO stimuli (id, word, kind, abstraction_level, source_note,
                                     ready, published, creator_session, created_at)
                VALUES (?, ?, 'custom', NULL, '', 0, 0, ?, ?)
                """,
                (sid, word.strip(), session_id, _now()),
            )
        return Stimulus(
            id=sid, word=word.strip(), source=StimulusSource.mine, abstraction_level=None,
            response_count=0, ready=False,
        )

    def mark_ready(self, stimulus_id: str) -> None:
        with self._conn() as con:
            con.execute("UPDATE stimuli SET ready = 1 WHERE id = ?", (stimulus_id,))

    def publish_stimulus(self, stimulus_id: str, session_id: str) -> Stimulus | None:
        """公開自訂詞到社群；只有建立者可以。不是建立者、不存在或是預設詞時回傳 None。"""
        with self._conn() as con:
            cur = con.execute(
                "UPDATE stimuli SET published = 1 WHERE id = ? AND kind = 'custom'"
                " AND creator_session = ?",
                (stimulus_id, session_id),
            )
            if cur.rowcount == 0:
                return None
        return self.get_stimulus(stimulus_id, session_id)

    def count_created_today(self, session_id: str) -> int:
        """該 session 今天（UTC）新增了幾個自訂詞。"""
        today = datetime.now(timezone.utc).date().isoformat()
        with self._conn() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM stimuli WHERE kind = 'custom' AND creator_session = ?"
                " AND substr(created_at, 1, 10) = ?",
                (session_id, today),
            ).fetchone()
        return int(row[0])

    # ------------------------------------------------------------------
    # 種子與投影器
    # ------------------------------------------------------------------

    def add_seed_points(
        self,
        stimulus_id: str,
        texts: list[str],
        vecs: np.ndarray,
        distances: np.ndarray | None = None,
    ) -> None:
        """加入 AI 種子點。distances 可省略：省略時在 cloud_points 以投影器的刺激詞向量現算。"""
        vecs = np.asarray(vecs)
        if texts and (vecs.ndim != 2 or vecs.shape[0] != len(texts)):
            raise ValueError("vecs 數量必須與 texts 相同")
        if distances is not None and len(distances) != len(texts):
            raise ValueError("distances 長度必須與 texts 相同")
        now = _now()
        with self._conn() as con:
            con.executemany(
                "INSERT INTO seeds (stimulus_id, text, vec, distance, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        stimulus_id,
                        t,
                        _to_blob(vecs[i]),
                        None if distances is None else float(distances[i]),
                        now,
                    )
                    for i, t in enumerate(texts)
                ],
            )

    def seed_vectors(self, stimulus_id: str) -> np.ndarray | None:
        """回傳該刺激詞所有 AI 種子的向量；沒有種子回傳 None。"""
        with self._conn() as con:
            rows = con.execute("SELECT vec FROM seeds WHERE stimulus_id = ? ORDER BY id", (stimulus_id,)).fetchall()
        if not rows:
            return None
        return np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])

    def save_projector(self, stimulus_id: str, data: dict) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO projectors (stimulus_id, data, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(stimulus_id) DO UPDATE SET data = excluded.data,"
                " updated_at = excluded.updated_at",
                (stimulus_id, json.dumps(data), _now()),
            )

    def load_projector(self, stimulus_id: str) -> dict | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT data FROM projectors WHERE stimulus_id = ?", (stimulus_id,)
            ).fetchone()
        return json.loads(row["data"]) if row else None

    # ------------------------------------------------------------------
    # 作答
    # ------------------------------------------------------------------

    def save_submission(
        self,
        *,
        stimulus_id: str,
        session_id: str,
        response: AnalysisResponse,
        vecs: np.ndarray,
    ) -> str:
        """儲存一次作答（shared=False），回傳新的 submission_id（uuid4 hex）。

        vecs 必須與 response.associations 一一對應；每則答案的 distance、weight
        取自 response.associations。存下的 response JSON 會改用新的 submission_id。
        """
        assocs = response.associations
        vecs = np.asarray(vecs)
        if vecs.ndim == 1 and len(assocs) == 1:
            vecs = vecs[None, :]
        if len(assocs) and (vecs.ndim != 2 or vecs.shape[0] != len(assocs)):
            raise ValueError("vecs 數量必須與 response.associations 相同")
        sub_id = uuid.uuid4().hex
        stored = response.model_copy(update={"submission_id": sub_id})
        fp = {k: getattr(response.fingerprint, k).raw for k in _FP_KEYS}
        with self._conn() as con:
            con.execute(
                "INSERT INTO submissions (id, stimulus_id, session_id, shared, created_at,"
                " fingerprint_raw, response_json) VALUES (?, ?, ?, 0, ?, ?, ?)",
                (sub_id, stimulus_id, session_id, _now(), json.dumps(fp),
                 stored.model_dump_json()),
            )
            if assocs:
                con.executemany(
                    "INSERT INTO answers (submission_id, idx, text, weight, distance, vec)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (sub_id, a.index, a.text, float(a.weight), float(a.distance),
                         _to_blob(vecs[i]))
                        for i, a in enumerate(assocs)
                    ],
                )
        return sub_id

    def share_submission(self, submission_id: str, session_id: str) -> bool:
        """同意分享：只有作答本人（session 相同）可以。成功（含已分享過）回傳 True。"""
        with self._conn() as con:
            cur = con.execute(
                "UPDATE submissions SET shared = 1, shared_at = COALESCE(shared_at, ?)"
                " WHERE id = ? AND session_id = ?",
                (_now(), submission_id, session_id),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # 公共分布（只含已分享的人類答案、weight > 0）
    # ------------------------------------------------------------------

    def _shared_answers(self, con: sqlite3.Connection, stimulus_id: str) -> list[sqlite3.Row]:
        return con.execute(
            "SELECT a.text, a.weight, a.distance, a.vec, sub.session_id FROM answers a"
            " JOIN submissions sub ON sub.id = a.submission_id"
            " WHERE sub.stimulus_id = ? AND sub.shared = 1 AND a.weight > 0"
            " ORDER BY sub.created_at, a.idx",
            (stimulus_id,),
        ).fetchall()

    def pool_vectors_by_user(self, stimulus_id: str) -> dict[str, np.ndarray]:
        """{session_id: 該使用者所有已分享、合理答案的向量 (m, dim)}。"""
        with self._conn() as con:
            rows = self._shared_answers(con, stimulus_id)
        grouped: dict[str, list[np.ndarray]] = {}
        for r in rows:
            grouped.setdefault(r["session_id"], []).append(_from_blob(r["vec"]))
        return {k: np.vstack(v) for k, v in grouped.items()}

    def pool_distances(self, stimulus_id: str) -> np.ndarray:
        with self._conn() as con:
            rows = self._shared_answers(con, stimulus_id)
        return np.array([r["distance"] for r in rows], dtype=np.float64)

    def pool_raw_fingerprints(self, stimulus_id: str, exclude_session: str | None) -> list[dict]:
        """已分享使用者的指紋原始值；同一 session 只取最新一次分享。"""
        with self._conn() as con:
            rows = con.execute(
                "SELECT session_id, fingerprint_raw FROM submissions"
                " WHERE stimulus_id = ? AND shared = 1 ORDER BY created_at",
                (stimulus_id,),
            ).fetchall()
        latest: dict[str, dict] = {}
        for r in rows:
            if exclude_session is not None and r["session_id"] == exclude_session:
                continue
            latest[r["session_id"]] = json.loads(r["fingerprint_raw"])
        return list(latest.values())

    def cloud_points(self, stimulus_id: str, limit: int = 400) -> list[CloudPoint]:
        """背景星雲：已分享的人類合理答案 + AI 種子，相同文字合併並計數。

        沒有存投影器時無法算座標，回傳空清單。排序：count 多的優先，同數時人類優先。
        """
        proj_data = self.load_projector(stimulus_id)
        if proj_data is None:
            return []
        proj = Projector.from_dict(proj_data)
        stim_unit = proj.stim_vec / max(float(np.linalg.norm(proj.stim_vec)), 1e-12)

        with self._conn() as con:
            human_rows = self._shared_answers(con, stimulus_id)
            seed_rows = con.execute(
                "SELECT text, vec, distance FROM seeds WHERE stimulus_id = ? ORDER BY id",
                (stimulus_id,),
            ).fetchall()

        # (source, text) -> [vec, distance, 計數集合或次數]
        groups: dict[tuple[str, str], dict] = {}
        for r in human_rows:
            g = groups.setdefault(
                ("human", r["text"]),
                {"vec": _from_blob(r["vec"]), "distance": r["distance"], "who": set()},
            )
            g["who"].add(r["session_id"])
        for r in seed_rows:
            vec = _from_blob(r["vec"])
            dist = r["distance"]
            if dist is None:
                cos = float(vec @ stim_unit) / max(float(np.linalg.norm(vec)), 1e-12)
                dist = min(max(1.0 - cos, 0.0), 2.0)
            g = groups.setdefault(("seed", r["text"]), {"vec": vec, "distance": dist, "n": 0})
            g["n"] += 1

        items = []
        for (source, text), g in groups.items():
            count = len(g["who"]) if source == "human" else g["n"]
            items.append((source, text, g, count))
        items.sort(key=lambda it: (-it[3], it[0] != "human", it[1]))

        return [
            CloudPoint(
                text=text,
                position=proj.project(g["vec"], g["distance"]),
                source=source,
                count=count,
            )
            for source, text, g, count in items[: max(limit, 0)]
        ]

    def pool_stats(self, stimulus_id: str, min_users: int) -> PoolStats:
        with self._conn() as con:
            users = con.execute(
                "SELECT COUNT(DISTINCT session_id) FROM submissions"
                " WHERE stimulus_id = ? AND shared = 1",
                (stimulus_id,),
            ).fetchone()[0]
            seeds = con.execute(
                "SELECT COUNT(*) FROM seeds WHERE stimulus_id = ?", (stimulus_id,)
            ).fetchone()[0]
        return PoolStats(
            stimulus_id=stimulus_id,
            human_users=int(users),
            seed_points=int(seeds),
            enough_for_percentile=int(users) >= min_users,
        )

