"""Voice API router — pure STT and TTS endpoints.

VoiceService is an internal utility called BY AIGateway.
It has no auth, no LLM calls, and no agent awareness.
AIGateway owns all orchestration and agent authentication.

Endpoints:
  POST /stt        raw WAV bytes → transcript JSON
  POST /tts        {text, voice, speed, ...} → base64 WAV + visemes
  POST /tts/raw    {text, voice, speed}      → raw WAV bytes
  GET  /voices     list available voices
  GET  /health     service status
"""
import asyncio
import base64
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..config import settings
from ..services import stt as stt_svc
from ..services.pipeline import buffer_bytes
from ..services.tts_glados import GladosTTS, get_glados, is_loaded as glados_loaded
from ..services.tts_piper import AtlasTTS, get_atlas, is_loaded as atlas_loaded

router = APIRouter(tags=["Voice"])
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: str = Field("glados", description="'glados' or 'atlas'")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speaking speed multiplier")
    # VITS expressiveness params (GLaDOS only — ignored for ATLAS)
    noise_scale: float = Field(0.333, ge=0.0, le=1.0, description="Phoneme variation (expressiveness)")
    noise_w: float     = Field(0.333, ge=0.0, le=1.0, description="Duration variation")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_engine(voice: str) -> GladosTTS | AtlasTTS:
    if voice.lower() == "atlas":
        return get_atlas()
    return get_glados()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/stt", summary="Audio → transcript")
async def stt_endpoint(request: Request):
    """
    Send raw WAV bytes (PCM 16-bit 16 kHz mono) in the request body.
    Returns transcribed text.  No authentication required.

    ESP32 example:
        POST /stt
        Content-Type: audio/wav
        [raw WAV bytes]
    """
    wav_bytes = await request.body()
    if not wav_bytes:
        raise HTTPException(400, "Empty body — send raw WAV bytes.")
    return await asyncio.to_thread(stt_svc.transcribe, wav_bytes)


@router.post("/tts", summary="Text → audio (JSON response with base64 WAV)")
async def tts_json_endpoint(req: TTSRequest):
    """
    Synthesise speech from text.  Returns JSON with base64-encoded WAV,
    a viseme timeline, and a buffer hint for ESP32 WiFi pre-buffering.

    The `noise_scale` and `noise_w` params let AIGateway pass emotional state
    through to the VITS synthesiser (GLaDOS voice only):
      - higher noise_scale → more expressive / varied delivery
      - lower  noise_scale → flat, robotic delivery
    """
    engine = _get_engine(req.voice)

    # Pass VITS params through to GLaDOS if supported
    kwargs = {"speed": req.speed}
    if req.voice.lower() == "glados":
        kwargs["noise_scale"] = req.noise_scale
        kwargs["noise_w"]     = req.noise_w

    result = await asyncio.to_thread(engine.synthesize, req.text, **kwargs)

    return {
        "voice":        req.voice,
        "audio":        base64.b64encode(result["audio"]).decode(),
        "audio_format": "wav",
        "sample_rate":  engine.sample_rate,
        "duration_ms":  result["duration_ms"],
        "buffer_bytes": buffer_bytes(engine.sample_rate, settings.buffer_hint_ms),
        "visemes":      result["visemes"],
    }


@router.post("/tts/raw", summary="Text → audio (raw WAV bytes)")
async def tts_raw_endpoint(req: TTSRequest):
    """Returns WAV directly — useful for curl testing or non-ESP32 clients."""
    engine = _get_engine(req.voice)
    result = await asyncio.to_thread(engine.synthesize, req.text, req.speed)
    return Response(
        content=result["audio"],
        media_type="audio/wav",
        headers={"X-Duration-Ms": str(result["duration_ms"])},
    )


@router.get("/voices", summary="List available TTS voices")
async def list_voices():
    return {
        "voices": [
            {
                "id":          "glados",
                "name":        "GLaDOS",
                "character":   "Aperture Science AI (Portal)",
                "description": "ONNX VITS — precise, condescending, darkly humorous",
                "sample_rate": 22050,
                "loaded":      glados_loaded(),
                "params":      ["speed", "noise_scale", "noise_w"],
            },
            {
                "id":          "atlas",
                "name":        "ATLAS",
                "character":   "Cooperative android (Portal 2)",
                "description": "Piper en_US-ryan-high — clear, professional AI-assistant",
                "sample_rate": 22050,
                "loaded":      atlas_loaded(),
                "params":      ["speed"],
            },
        ],
        "default": settings.default_voice,
    }


@router.get("/health", summary="Service health")
async def health():
    return {
        "status":        "ok",
        "service":       "VoiceService",
        "port":          settings.port,
        "stt_loaded":    stt_svc.is_loaded(),
        "glados_loaded": glados_loaded(),
        "atlas_loaded":  atlas_loaded(),
        "note":          "Internal utility — called by AIGateway, no auth required.",
    }
