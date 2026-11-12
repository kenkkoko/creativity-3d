"""創造力 3D 的資料契約（Pydantic models）。

所有 API 輸入輸出、LLM 的 Structured Output 都以這裡為準。
`scripts/export_schema.py` 會把它們匯出成 `schema.json`。

命名慣例：
- *Request / *Response：前端 ↔ FastAPI
- Llm*：交給 Ollama `format=` 的 JSON Schema（Structured Output）
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# ---------------------------------------------------------------------------
# 共用型別與常數
# ---------------------------------------------------------------------------

MAX_ASSOCIATIONS = 60  # 一次作答最多幾則聯想
MAX_TEXT_LEN = 20  # 每則聯想最多幾個字
MAX_NOTE_LEN = 60  # 選填說明最多幾個字

AssociationText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LEN)
]
NoteText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=MAX_NOTE_LEN)]
Score01 = Annotated[float, Field(ge=0.0, le=1.0)]
Percentile = Annotated[float, Field(ge=0.0, le=100.0)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# 刺激詞
# ---------------------------------------------------------------------------


class StimulusSource(str, Enum):
    preset = "preset"  # 預設詞庫
    community = "community"  # 使用者新增且已公開
    mine = "mine"  # 使用者新增、尚未公開（只有建立者看得到）


StimulusWord = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=6, pattern=r"^[\u4e00-\u9fff]+$")
]


class Stimulus(Strict):
    id: str = Field(description="刺激詞代號，例如 s05")
    word: str = Field(description="中文刺激詞，例如 藝術")
    source: StimulusSource
    abstraction_level: int | None = Field(
        default=None, ge=1, le=10, description="1=最抽象，10=最具象；自訂詞為 null（未分級）"
    )
    source_note: str = Field(default="", description="抽象程度的依據說明")
    response_count: int = Field(default=0, ge=0, description="已同意分享的作答人數")
    ready: bool = Field(default=True, description="AI 種子產生完成才為 true")


class CreateStimulusRequest(Strict):
    word: StimulusWord
    client_session_id: str = Field(min_length=8, max_length=64)


class PublishStimulusRequest(Strict):
    client_session_id: str = Field(min_length=8, max_length=64)
    confirm_publish: bool = Field(description="必須由建立者在介面上明確勾選")


class LlmStimulusCheck(Strict):
    """LLM 判斷自訂詞是否適合當刺激詞。"""

    is_associable: bool = Field(description="是否為一般人能自由聯想的詞")
    reason: Annotated[str, StringConstraints(max_length=60)]


class LlmSeedBatch(Strict):
    """LLM 產生的 AI 種子聯想。"""

    associations: list[Annotated[str, StringConstraints(min_length=1, max_length=MAX_TEXT_LEN)]] = Field(
        min_length=10, max_length=40
    )


# ---------------------------------------------------------------------------
# 作答輸入
# ---------------------------------------------------------------------------


class AssociationInput(Strict):
    text: AssociationText = Field(description="使用者寫下的聯想（中文）")
    note: NoteText | None = Field(default=None, description="選填：我怎麼想到的")


class SubmitRequest(Strict):
    stimulus_id: str
    associations: list[AssociationInput] = Field(min_length=1, max_length=MAX_ASSOCIATIONS)
    elapsed_seconds: int = Field(ge=0, le=600, description="實際作答秒數")
    client_session_id: str = Field(
        min_length=8, max_length=64, description="瀏覽器產生的匿名 ID，不含個資"
    )


# ---------------------------------------------------------------------------
# 合理性檢查（三層）
# ---------------------------------------------------------------------------


class RuleFlag(str, Enum):
    ok = "ok"
    empty = "empty"
    no_cjk = "no_cjk"  # 不含中文字
    gibberish = "gibberish"  # 亂碼、注音亂打、同字連打
    duplicate = "duplicate"  # 和自己前面的答案重複
    same_as_stimulus = "same_as_stimulus"
    blocked_word = "blocked_word"  # 不雅字詞
    injection = "injection"  # 疑似 prompt injection
    too_long = "too_long"


class Verdict(str, Enum):
    reasonable = "reasonable"  # 合理 → 權重 1.0
    borderline = "borderline"  # 勉強說得通 → 權重 0.5
    unreasonable = "unreasonable"  # 找不到聯想橋 → 權重 0.0


VERDICT_WEIGHT: dict[Verdict, float] = {
    Verdict.reasonable: 1.0,
    Verdict.borderline: 0.5,
    Verdict.unreasonable: 0.0,
}


class LlmJudgement(Strict):
    """LLM 對「單一聯想」的判斷。交給 Ollama 做 Structured Output。"""

    index: int = Field(ge=0, description="對應輸入清單中的位置")
    verdict: Verdict
    bridge: list[Annotated[str, StringConstraints(max_length=12)]] = Field(
        max_length=3,
        description="從刺激詞走到聯想的中間概念，0～3 個；verdict=unreasonable 時必須為空",
    )
    reason: Annotated[str, StringConstraints(max_length=30)] = Field(
        description="一句話理由，繁體中文，30 字以內"
    )


class LlmJudgementBatch(Strict):
    """一次判斷多則聯想（減少 LLM 呼叫次數）。"""

    judgements: list[LlmJudgement]


# ---------------------------------------------------------------------------
# 計分結果
# ---------------------------------------------------------------------------


class Point3D(Strict):
    x: float
    y: float
    z: float


class ScoredAssociation(Strict):
    index: int
    text: str
    note: str | None = None
    rule_flag: RuleFlag
    verdict: Verdict
    weight: Score01 = Field(description="合理性權重：1.0 / 0.5 / 0")
    distance: float = Field(ge=0.0, le=2.0, description="語義距離 = 1 - cosine 相似度")
    effective_distance: float = Field(ge=0.0, le=2.0, description="distance × weight")
    rarity: Score01 | None = Field(default=None, description="越接近 1 越少人想到；資料不足時為 null")
    outlier: bool = Field(description="距離是否為極端值（向量層標記）")
    bridge: list[str] = Field(default_factory=list)
    bridge_positions: list[Point3D] = Field(
        default_factory=list, description="聯想橋中間概念的 3D 座標，與 bridge 一一對應"
    )
    reason: str = ""
    position: Point3D = Field(description="3D 座標；半徑 = 真實語義距離，方向由投影決定")


class DimensionScore(Strict):
    raw: float | None = Field(description="原始值；資料不足時為 null")
    percentile: Percentile | None = Field(
        default=None, description="在同一刺激詞所有使用者中的百分位；樣本不足時為 null"
    )


class Fingerprint(Strict):
    """創造力指紋：五個維度，不合成單一總分。"""

    distance: DimensionScore = Field(description="想得多遠：有效距離平均")
    breadth: DimensionScore = Field(description="想得多散：合理答案彼此的平均距離")
    rarity: DimensionScore = Field(description="多少人也想到：稀有度平均")
    appropriateness: DimensionScore = Field(description="合不合理：合理權重平均")
    fluency: DimensionScore = Field(description="想了多少：有效答案數")


class LlmInterpretation(Strict):
    """LLM 寫的解讀。只能引用 Fingerprint 裡的數字。"""

    headline: Annotated[str, StringConstraints(max_length=30)] = Field(
        description="一句話標題，例如「遠跳型探險家」"
    )
    strengths: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=4, max_length=60)]] = Field(
        min_length=1, max_length=3
    )
    suggestions: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=4, max_length=60)]] = Field(
        min_length=1, max_length=3
    )
    cited_numbers: list[float] = Field(
        description="解讀中引用到的所有數字，系統會逐一比對 Fingerprint，不符就重試"
    )


class PoolStats(Strict):
    stimulus_id: str
    human_users: int = Field(ge=0, description="已同意加入分布的使用者數")
    seed_points: int = Field(ge=0, description="AI 種子點數量")
    enough_for_percentile: bool = Field(description="human_users ≥ 門檻（預設 20）才顯示百分位")


class CloudPoint(Strict):
    text: str
    position: Point3D
    source: str = Field(pattern="^(human|seed)$")
    count: int = Field(ge=1, description="相同或極相近答案被幾個人寫過")


class AnalysisResponse(Strict):
    submission_id: str
    stimulus: Stimulus
    associations: list[ScoredAssociation]
    fingerprint: Fingerprint
    interpretation: LlmInterpretation | None = Field(
        description="LLM 失敗 3 次時為 null，前端改顯示預設文字"
    )
    pool: PoolStats
    cloud: list[CloudPoint] = Field(description="其他使用者與 AI 種子的點，用來畫背景星雲")
    warnings: list[str] = Field(default_factory=list, description="例如：有 3 則被判定不合理")
    created_at: datetime


# ---------------------------------------------------------------------------
# 寫入公共分布（需使用者確認）
# ---------------------------------------------------------------------------


class ConsentRequest(Strict):
    submission_id: str
    agree_to_share: bool = Field(description="必須由使用者在介面上明確勾選")


class ConsentResponse(Strict):
    submission_id: str
    shared: bool
    message: str


# ---------------------------------------------------------------------------
# 玩家身分與紀錄
# ---------------------------------------------------------------------------

Nickname = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12,
                           pattern=r"^[\u4e00-\u9fffA-Za-z0-9_]+$")
]


class PlayerLoginRequest(Strict):
    nickname: Nickname = Field(description="玩家暱稱，1–12 字，中英數與底線")
    pin: str = Field(pattern=r"^\d{4}$", description="4 位數字 PIN")
    client_session_id: str = Field(min_length=8, max_length=64, description="目前瀏覽器的匿名 ID")


class PlayerLoginResponse(Strict):
    nickname: str
    session_id: str = Field(description="之後所有 API 都改用這個 ID")
    is_new: bool = Field(description="true = 第一次建立身分")


class PlayerInfo(Strict):
    nickname: str | None


class HistoryItem(Strict):
    submission_id: str
    stimulus_id: str
    stimulus_word: str
    created_at: datetime
    fluency: float | None
    distance: float | None
    headline: str | None
    shared: bool


# ---------------------------------------------------------------------------
# 錯誤
# ---------------------------------------------------------------------------


class ErrorCode(str, Enum):
    invalid_input = "invalid_input"
    blocked_input = "blocked_input"
    auth_failed = "auth_failed"
    rate_limited = "rate_limited"
    unknown_stimulus = "unknown_stimulus"
    llm_unavailable = "llm_unavailable"
    internal = "internal"


class ErrorResponse(Strict):
    code: ErrorCode
    message: str = Field(description="給使用者看的繁體中文訊息")


ALL_MODELS: list[type[BaseModel]] = [
    Stimulus,
    CreateStimulusRequest,
    PublishStimulusRequest,
    LlmStimulusCheck,
    LlmSeedBatch,
    SubmitRequest,
    LlmJudgementBatch,
    LlmInterpretation,
    AnalysisResponse,
    ConsentRequest,
    ConsentResponse,
    PlayerLoginRequest,
    PlayerLoginResponse,
    PlayerInfo,
    HistoryItem,
    ErrorResponse,
]
