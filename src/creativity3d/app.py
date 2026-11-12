"""FastAPI 應用程式。

階段 3 只有骨架：健康檢查、預設刺激詞、靜態前端。
作答、計分、同意寫入等端點在階段 5 加入。
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import PROJECT_ROOT, Settings, get_settings
from .logging_setup import setup_logging
from .providers import build_providers
from .schemas import ErrorCode, ErrorResponse, Stimulus, StimulusSource

log = logging.getLogger(__name__)
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def load_preset_stimuli() -> list[Stimulus]:
    raw = resources.files("creativity3d.data").joinpath("stimuli_preset.json").read_text("utf-8")
    return [Stimulus(source=StimulusSource.preset, **item) for item in json.loads(raw)]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_dir)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        app.state.settings = settings
        app.state.llm, app.state.embedder = build_providers(settings)
        app.state.preset = load_preset_stimuli()
        log.info("啟動完成：provider=%s llm=%s embed=%s", settings.provider, settings.llm_model, settings.embed_model)
        yield

    app = FastAPI(title="創造力 3D", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("未處理的錯誤：%s", exc)
        body = ErrorResponse(code=ErrorCode.internal, message="系統發生錯誤，請稍後再試。")
        return JSONResponse(status_code=500, content=body.model_dump())

    @app.get("/api/health")
    def health(request: Request) -> dict:
        s: Settings = request.app.state.settings
        llm_status = request.app.state.llm.health()
        installed = set(llm_status.get("installed_models", []))
        missing = [
            m for m in (s.llm_model, s.embed_model)
            if s.provider == "ollama" and llm_status.get("ok")
            and m not in installed and f"{m}:latest" not in installed
        ]
        return {
            "ok": bool(llm_status.get("ok")) and not missing,
            "provider": s.provider,
            "llm_model": s.llm_model,
            "embed_model": s.embed_model,
            "missing_models": missing,
            "detail": llm_status,
        }

    @app.get("/api/stimuli", response_model=list[Stimulus])
    def stimuli(request: Request) -> list[Stimulus]:
        return request.app.state.preset

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    return app
