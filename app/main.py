"""VoiceService — FastAPI entry point.

Standalone microservice providing STT, TTS (GLaDOS + ATLAS), and a full
voice-chat pipeline for ESP32-S3 clients and AIGateway integration.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers.voice import router as voice_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VoiceService starting on %s:%d", settings.host, settings.port)
    logger.info("AIGateway endpoint: %s", settings.aigateway_url)

    # Warm-up: pre-load models so the first real request isn't slow.
    # Each loader is wrapped in try/except so a missing model file doesn't
    # prevent the service from starting.
    import asyncio
    async def _warm(name: str, fn):
        try:
            await asyncio.to_thread(fn)
            logger.info("✓ %s ready", name)
        except Exception as exc:
            logger.warning("✗ %s not ready — %s  (run download_models.py)", name, exc)

    from .services import stt as stt_svc
    from .services.tts_glados import get_glados
    from .services.tts_piper import get_atlas

    await _warm("Whisper STT",  stt_svc.get_model)
    await _warm("GLaDOS TTS",   lambda: get_glados().synthesize("test", 1.0))
    await _warm("ATLAS TTS",    lambda: get_atlas().synthesize("test", 1.0))

    yield

    logger.info("VoiceService shutting down.")


app = FastAPI(
    title="VoiceService",
    description=(
        "STT/TTS voice pipeline for ESP32-S3 + AIGateway integration.\n\n"
        "**Voices:** GLaDOS (Portal ONNX VITS) · ATLAS (Piper en_US-ryan-high)\n\n"
        "**STT:** faster-whisper (base.en)\n\n"
        "**Pipeline:** audio → Whisper → AIGateway LLM → TTS → audio + visemes + actions"
    ),
    version="1.0.0",
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
