"""FastAPI 應用程式：API 端點與靜態前端。"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline
from .config import PROJECT_ROOT, Settings, get_settings
from .logging_setup import setup_logging
from .pipeline import AppError, Services
from .providers import build_providers
from .schemas import (
    AnalysisResponse,
    ConsentRequest,
    ConsentResponse,
    CreateStimulusRequest,
    ErrorCode,
    ErrorResponse,
    PublishStimulusRequest,
    Stimulus,
    StimulusSource,
    SubmitRequest,
)
from .store import Store

log = logging.getLogger(__name__)
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def load_preset_stimuli() -> list[Stimulus]:
    raw = resources.files("creativity3d.data").joinpath("stimuli_preset.json").read_text("utf-8")
    return [Stimulus(source=StimulusSource.preset, **item) for item in json.loads(raw)]


def _error(code: ErrorCode, message: str, status: int) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorResponse(code=code, message=message).model_dump())


def create_app(settings: Settings | None = None, *, seed_on_startup: bool = True,
               services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_dir)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        if services is not None:
            svc = services
        else:
            llm, embedder = build_providers(settings)
            svc = Services(settings, llm, embedder, Store(settings.data_dir / "creativity3d.sqlite"))
        svc.store.ensure_presets(load_preset_stimuli())
        app.state.svc = svc
        if seed_on_startup:
            threading.Thread(target=pipeline.seed_missing_presets, args=(svc,), daemon=True).start()
        log.info("啟動完成：provider=%s llm=%s embed=%s", settings.provider, settings.llm_model, settings.embed_model)
        yield

    app = FastAPI(title="創造力 3D", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(AppError)
    async def app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error(exc.code, exc.message, exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        log.info("輸入格式錯誤：%s %s", request.url.path, exc.errors()[:3])
        if request.url.path == "/api/stimuli":
            msg = "刺激詞需為 1–6 個中文字，不含英文、數字或符號。"
        elif request.url.path == "/api/submit":
            msg = "輸入格式不正確：每則聯想 1–20 字、最多 60 則，說明最多 60 字。"
        else:
            msg = "輸入格式不正確。"
        return _error(ErrorCode.invalid_input, msg, 422)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("未處理的錯誤：%s", exc)
        return _error(ErrorCode.internal, "系統發生錯誤，請稍後再試。", 500)

    def svc_of(request: Request) -> Services:
        return request.app.state.svc

    @app.get("/api/health")
    def health(request: Request) -> dict:
        svc = svc_of(request)
        s = svc.settings
        st = svc.llm.health()
        installed = set(st.get("installed_models", []))
        missing = [
            m for m in (s.llm_model, s.embed_model)
            if s.provider == "ollama" and st.get("ok") and m not in installed and f"{m}:latest" not in installed
        ]
        return {"ok": bool(st.get("ok")) and not missing, "provider": s.provider,
                "llm_model": s.llm_model, "embed_model": s.embed_model,
                "missing_models": missing, "detail": st}

    @app.get("/api/config")
    def config(request: Request) -> dict:
        s = svc_of(request).settings
        return {"answer_seconds": s.answer_seconds, "percentile_min_users": s.percentile_min_users}

    @app.get("/api/stimuli", response_model=list[Stimulus])
    def list_stimuli(request: Request, client_session_id: str | None = Query(default=None)) -> list[Stimulus]:
        return svc_of(request).store.list_stimuli(client_session_id)

    @app.post("/api/stimuli", response_model=Stimulus)
    def create_stimulus(body: CreateStimulusRequest, request: Request, bg: BackgroundTasks) -> Stimulus:
        svc = svc_of(request)
        stim, needs_seed = pipeline.create_stimulus(svc, body.word, body.client_session_id)
        if needs_seed:
            bg.add_task(pipeline.seed_stimulus, svc, stim)
        return stim

    @app.post("/api/stimuli/{stimulus_id}/publish", response_model=Stimulus)
    def publish(stimulus_id: str, body: PublishStimulusRequest, request: Request) -> Stimulus:
        if not body.confirm_publish:
            raise AppError(ErrorCode.invalid_input, "需要勾選確認才能公開。")
        stim = svc_of(request).store.publish_stimulus(stimulus_id, body.client_session_id)
        if stim is None:
            raise AppError(ErrorCode.unknown_stimulus, "找不到這個詞，或你不是建立者。", 404)
        return stim

    @app.post("/api/submit", response_model=AnalysisResponse)
    def submit(body: SubmitRequest, request: Request) -> AnalysisResponse:
        return pipeline.analyze(svc_of(request), body)

    @app.post("/api/consent", response_model=ConsentResponse)
    def consent(body: ConsentRequest, request: Request,
                client_session_id: str = Query(min_length=8, max_length=64)) -> ConsentResponse:
        if not body.agree_to_share:
            return ConsentResponse(submission_id=body.submission_id, shared=False, message="沒有勾選同意，答案不會加入星圖。")
        ok = svc_of(request).store.share_submission(body.submission_id, client_session_id)
        if not ok:
            raise AppError(ErrorCode.invalid_input, "找不到這次作答，或已經分享過。", 404)
        return ConsentResponse(submission_id=body.submission_id, shared=True, message="已匿名加入星圖，謝謝你！")

    @app.get("/api/cloud/{stimulus_id}")
    def cloud(stimulus_id: str, request: Request, client_session_id: str | None = Query(default=None)) -> dict:
        svc = svc_of(request)
        if svc.store.get_stimulus(stimulus_id, client_session_id) is None:
            raise AppError(ErrorCode.unknown_stimulus, "找不到這個刺激詞。", 404)
        return {"points": [p.model_dump() for p in svc.store.cloud_points(stimulus_id)],
                "pool": svc.store.pool_stats(stimulus_id, svc.settings.percentile_min_users).model_dump()}

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    return app
