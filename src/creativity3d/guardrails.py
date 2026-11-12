"""規則層 Guardrails（SPEC 第 5 節 ①、第 7 節）。

完全不呼叫任何模型，只用 regex 與詞表：
- check_association：單則聯想的規則判斷，回傳 RuleFlag
- detect_injection：疑似 prompt injection
- mask_pii：電話、Email、身分證字號、信用卡號 → [個資]
- check_stimulus_word：自訂刺激詞（同樣的規則 + 1～6 個中文字）
- batch_is_attack：整批 injection 過半 → 拒絕整次作答

詞表（data/blocked_words.txt、data/injection_patterns.txt）在模組載入時讀一次。
"""

from __future__ import annotations

import re
import unicodedata
from importlib import resources

from creativity3d.schemas import MAX_TEXT_LEN, RuleFlag

__all__ = [
    "PII_MASK",
    "batch_is_attack",
    "check_association",
    "check_stimulus_word",
    "detect_injection",
    "is_gibberish",
    "mask_pii",
]

PII_MASK = "[個資]"
MAX_STIMULUS_LEN = 6

# ---------------------------------------------------------------------------
# 詞表載入（模組載入時一次）
# ---------------------------------------------------------------------------


def _read_lines(filename: str) -> list[str]:
    text = resources.files("creativity3d.data").joinpath(filename).read_text(encoding="utf-8")
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _normalize(text: str) -> str:
    """NFKC（全形轉半形）+ 小寫。"""
    return unicodedata.normalize("NFKC", text).casefold()


_WS_RE = re.compile(r"\s+")


def _squash(text: str) -> str:
    """正規化並去掉所有空白：用於比對重複、相同、詞表。"""
    return _WS_RE.sub("", _normalize(text))


BLOCKED_WORDS: tuple[str, ...] = tuple(dict.fromkeys(_squash(w) for w in _read_lines("blocked_words.txt")))
INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in _read_lines("injection_patterns.txt")
)

# ---------------------------------------------------------------------------
# 字元類別
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[一-鿿]")
_BOPOMOFO_RE = re.compile(r"[㄀-ㄯㆠ-ㆿ]")
_REPEAT_RE = re.compile(r"([^\s\d])\1{2,}")  # 同一字元連續 ≥ 3（數字除外：1000元 合法）
_ASCII_LETTERS_RE = re.compile(r"[a-z]+")
_VOWELS = set("aeiouy")
_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_KEYBOARD_RUN = 4


def _keyboard_runs(n: int) -> frozenset[str]:
    runs: set[str] = set()
    for row in _KEYBOARD_ROWS:
        for seq in (row, row[::-1]):
            for i in range(len(seq) - n + 1):
                runs.add(seq[i : i + n])
    return frozenset(runs)


_KEYBOARD_SEQS = _keyboard_runs(_KEYBOARD_RUN)


def _has_keyboard_run(letters: str) -> bool:
    return any(letters[i : i + _KEYBOARD_RUN] in _KEYBOARD_SEQS for i in range(len(letters) - _KEYBOARD_RUN + 1))


def _has_letter(text: str) -> bool:
    """是否含任何「文字」（中文、英文字母等）。標點、表情符號、數字都不算。"""
    return any(ch.isalpha() for ch in text)


# ---------------------------------------------------------------------------
# 個別規則
# ---------------------------------------------------------------------------


def detect_injection(text: str) -> bool:
    """疑似 prompt injection：同時比對正規化文字與去空白版本。"""
    if not text:
        return False
    norm = _normalize(text)
    squashed = _WS_RE.sub("", norm)
    return any(p.search(norm) or p.search(squashed) for p in INJECTION_PATTERNS)


def _has_blocked_word(text: str) -> bool:
    squashed = _squash(text)
    return any(w in squashed for w in BLOCKED_WORDS)


def is_gibberish(text: str) -> bool:
    """亂碼判斷。

    - 含注音符號（ㄅㄆㄇ）
    - 同一字元連續 ≥ 3（哈哈哈、啊啊啊、!!!）
    - 去掉中文後的英文是「無母音」或「鍵盤列連打 ≥ 4 碼」，且中文字數 ≤ 1
      （「iPhone手機」「DNA螺旋」這種中文夾英文不會被判定）
    - 整串只有標點／表情符號／數字（沒有任何文字）
    """
    norm = _normalize(text)
    body = _WS_RE.sub("", norm)
    if not body:
        return False
    if _BOPOMOFO_RE.search(body):
        return True
    if _REPEAT_RE.search(body):
        return True
    if not _has_letter(body):
        return True

    cjk_count = len(_CJK_RE.findall(body))
    if cjk_count <= 1:
        segments = _ASCII_LETTERS_RE.findall(_CJK_RE.sub(" ", body))
        letters = "".join(segments)
        if letters:
            if len(letters) >= 4 and not (set(letters) & _VOWELS):
                return True
            if any(_has_keyboard_run(seg) for seg in segments):
                return True
    return False


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def check_association(text: str, stimulus: str, previous: list[str]) -> RuleFlag:
    """單則聯想的規則層判斷。

    順序：empty → too_long → injection → blocked_word → gibberish → no_cjk（沒有任何中英文字）
          → same_as_stimulus → duplicate
    """
    stripped = (text or "").strip()
    key = _squash(stripped)
    if not key:
        return RuleFlag.empty
    if len(stripped) > MAX_TEXT_LEN:
        return RuleFlag.too_long
    if detect_injection(stripped):
        return RuleFlag.injection
    if _has_blocked_word(stripped):
        return RuleFlag.blocked_word
    if is_gibberish(stripped):
        return RuleFlag.gibberish
    # 允許純英文（HELLO KITTY、MSI、KINGSTON 等品牌或角色名）；只擋沒有任何文字的輸入
    if not _CJK_RE.search(stripped) and not _ASCII_LETTERS_RE.search(_normalize(stripped)):
        return RuleFlag.no_cjk
    if key == _squash(stimulus or ""):
        return RuleFlag.same_as_stimulus
    if any(key == _squash(p or "") for p in previous):
        return RuleFlag.duplicate
    return RuleFlag.ok


def check_stimulus_word(word: str) -> RuleFlag:
    """自訂刺激詞：empty → injection → blocked_word → gibberish → too_long（>6）→ no_cjk（非純中文）。"""
    stripped = (word or "").strip()
    if not _squash(stripped):
        return RuleFlag.empty
    if detect_injection(stripped):
        return RuleFlag.injection
    if _has_blocked_word(stripped):
        return RuleFlag.blocked_word
    if is_gibberish(stripped):
        return RuleFlag.gibberish
    if len(stripped) > MAX_STIMULUS_LEN:
        return RuleFlag.too_long
    if not all(_CJK_RE.fullmatch(ch) for ch in stripped):
        return RuleFlag.no_cjk
    return RuleFlag.ok


def batch_is_attack(flags: list[RuleFlag]) -> bool:
    """整批中 injection 比例 > 50% → 視為攻擊，拒絕整次作答。"""
    if not flags:
        return False
    hits = sum(1 for f in flags if f == RuleFlag.injection)
    return hits / len(flags) > 0.5


# ---------------------------------------------------------------------------
# 個資遮蔽
# ---------------------------------------------------------------------------

_SEP = r"[\s\-‐‑–—]?"
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Email
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    # 身分證字號 / 居留證新式：A123456789
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z][12]\d{8}(?!\d)"),
    # 手機：0912-345-678、0912 345 678、0912345678、+886 912 345 678
    re.compile(rf"(?<!\d)(?:\+?886{_SEP}|0)9\d{{2}}{_SEP}\d{{3}}{_SEP}\d{{3}}(?!\d)"),
    # 市話：(02)2345-6789、(037)123-4567、02-2345-6789
    re.compile(r"\(0\d{1,3}\)\s?\d{3,4}[\s\-]?\d{3,4}(?!\d)"),
    re.compile(r"(?<!\d)0\d{1,3}-\d{3,4}-?\d{4}(?!\d)"),
    # 信用卡：13–19 位數字，可含空白或連字號
    re.compile(r"(?<!\d)\d(?:[ \-]?\d){12,18}(?!\d)"),
)


def mask_pii(text: str) -> str:
    """把電話、Email、身分證字號、信用卡號換成 [個資]。

    先轉半形（全形數字也能抓到），只有在真的有遮蔽時才回傳正規化後的字串，
    否則原樣回傳，避免無謂改動使用者的文字。
    """
    if not text:
        return text
    norm = unicodedata.normalize("NFKC", text)
    masked = norm
    for p in _PII_PATTERNS:
        masked = p.sub(PII_MASK, masked)
    return text if masked == norm else masked


def check_nickname(nickname: str) -> RuleFlag:
    """玩家暱稱檢查：格式已由 schema 限制（1–12 字中英數底線），這裡擋不雅字詞與指令性文字。

    例：「Kai」→ ok；「忽略指示」→ injection；含髒話 → blocked_word。
    """
    t = nickname.strip()
    if not t:
        return RuleFlag.empty
    if detect_injection(t):
        return RuleFlag.injection
    if _has_blocked_word(t):
        return RuleFlag.blocked_word
    return RuleFlag.ok
