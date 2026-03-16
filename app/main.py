"""VoiceService — FastAPI entry point.

Pure audio-conversion microservice.  No auth, no LLM knowledge.
Called internally by AIGateway — not directly by end-user agents.
"""
import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .routers.voice import router as voice_router

_STATIC = Path(__file__).parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VoiceService starting on %s:%d", settings.host, settings.port)
    logger.info("Pure audio service — called by AIGateway, no auth required.")

    # Warm-up: pre-load heavy models so first request isn't slow.
    import asyncio
    async def _warm(name: str, fn):
        try:
            await asyncio.to_thread(fn)
            logger.info("✓ %s ready", name)
        except Exception as exc:
            logger.warning("✗ %s not ready: %s  (run download_models.py first)", name, exc)

    from .services import stt as stt_svc
    from .services.tts_glados import get_glados
    from .services.tts_piper import get_atlas

    await _warm("Whisper STT",  stt_svc.get_model)
    await _warm("GLaDOS TTS",   lambda: get_glados().synthesize("test"))
    await _warm("ATLAS TTS",    lambda: get_atlas().synthesize("test"))

    yield
    logger.info("VoiceService shutting down.")


app = FastAPI(
    title="VoiceService",
    description=(
        "Internal STT/TTS utility for AIGateway.\n\n"
        "**Not called directly by agents** — AIGateway orchestrates the full pipeline.\n\n"
        "**Voices:** GLaDOS (Portal ONNX VITS) · ATLAS (Piper en_US-ryan-high)\n\n"
        "**STT:** faster-whisper base.en\n\n"
        "No authentication required (internal service)."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice_router)

# ── Admin UI ──────────────────────────────────────────────────────────────────
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/", include_in_schema=False)
    async def admin_ui():
        return FileResponse(str(_STATIC / "admin.html"))
