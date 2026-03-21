"""Voice API router — pure STT and TTS endpoints.

VoiceService is an internal utility called BY AIGateway.
It has no auth, no LLM calls, and no agent awareness.
AIGateway owns all orchestration and agent authentication.

Endpoints:
  POST /stt        raw WAV bytes → transcript JSON
  POST /tts        {text, voice, speed, ...} → base64 WAV + visemes
  POST /tts/raw    {text, voice, speed}      → raw WAV bytes
  GET  /voices     list available voices
  GET  /activity   current state + recent operation log
  GET  /health     service status
"""
import asyncio
import base64
import logging
import time
from collections import deque

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..config import settings
from ..services import stt as stt_svc
from ..services.pipeline import buffer_bytes
from ..services.tts_glados import GladosTTS, get_glados, is_loaded as glados_loaded
from ..services.tts_piper import (
    PiperTTS,
    get_atlas,
    get_jarvis,
    get_terminator,
    is_loaded as piper_loaded,
)
from ..services.tts_xtts import XttsTTS, get_tars, is_loaded as xtts_loaded

router = APIRouter(tags=["Voice"])
logger = logging.getLogger(__name__)

# ── Activity tracking ────────────────────────────────────────────────────────
_voice_state: dict = {"current": "idle", "since": None}
_voice_log: deque = deque(maxlen=200)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: str = Field("glados", description="'glados', 'atlas', 'jarvis', 'c3po', 'tars', or 'terminator'")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speaking speed multiplier")
    # VITS expressiveness params (GLaDOS only — ignored for ATLAS)
    noise_scale: float = Field(0.333, ge=0.0, le=1.0, description="Phoneme variation (expressiveness)")
    noise_w: float     = Field(0.333, ge=0.0, le=1.0, description="Duration variation")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_engine(voice: str) -> GladosTTS | PiperTTS | XttsTTS:
    v = voice.lower()
    if v in ("jarvis", "c3po"):
        return get_jarvis()
    if v == "tars":
        return get_tars()
    if v == "terminator":
        return get_terminator()
    if v == "atlas":
        return get_atlas()
    return get_glados()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/stt", summary="Audio → transcript")
async def stt_endpoint(request: Request):
    wav_bytes = await request.body()
    if not wav_bytes:
        raise HTTPException(400, "Empty body — send raw WAV bytes.")
    started = time.time()
    _voice_state.update({"current": "stt", "since": started})
    error = None
    result = {}
    try:
        result = await asyncio.to_thread(stt_svc.transcribe, wav_bytes)
        return result
    except Exception as e:
        error = str(e)
        raise
    finally:
        fin = time.time()
        _voice_state.update({"current": "idle", "since": None})
        transcript = result.get("text", "") if isinstance(result, dict) else ""
        _voice_log.appendleft({
            "id": int(started * 1000),
            "type": "stt",
            "started_at": started,
            "finished_at": fin,
            "duration_ms": int((fin - started) * 1000),
            "transcript": transcript,
            "transcript_words": len(transcript.split()) if transcript else 0,
            "audio_bytes": len(wav_bytes),
            "error": error,
        })


@router.post("/tts", summary="Text → audio (JSON response with base64 WAV)")
async def tts_json_endpoint(req: TTSRequest):
    engine = _get_engine(req.voice)
    kwargs = {"speed": req.speed}
    if req.voice.lower() == "glados":
        kwargs["noise_scale"] = req.noise_scale
        kwargs["noise_w"]     = req.noise_w

    started = time.time()
    _voice_state.update({"current": "tts", "since": started, "voice": req.voice, "text_preview": req.text[:60]})
    error = None
    result = {}
    try:
        result = await asyncio.to_thread(engine.synthesize, req.text, **kwargs)
        payload = {
            "voice":         req.voice,
            "audio":         base64.b64encode(result["audio"]).decode(),
            "audio_format":  "wav",
            "sample_rate":   engine.sample_rate,
            "duration_ms":   result["duration_ms"],
            "synthesis_ms":  int((time.time() - started) * 1000),
            "buffer_bytes":  buffer_bytes(engine.sample_rate, settings.buffer_hint_ms),
            "visemes":       result["visemes"],
        }
        return payload
    except Exception as e:
        error = str(e)
        raise
    finally:
        fin = time.time()
        _voice_state.update({"current": "idle", "since": None})
        syn_ms   = int((fin - started) * 1000)
        audio_ms = result.get("duration_ms") if result else None
        _voice_log.appendleft({
            "id": int(started * 1000),
            "type": "tts",
            "started_at": started,
            "finished_at": fin,
            "duration_ms": syn_ms,
            "synthesis_ms": syn_ms,
            "text": req.text,          # full text, not truncated
            "voice": req.voice,
            "speed": req.speed,
            "noise_scale": req.noise_scale if req.voice.lower() == "glados" else None,
            "noise_w":     req.noise_w     if req.voice.lower() == "glados" else None,
            "audio_ms":    audio_ms,
            "viseme_count": len(result.get("visemes", [])) if result else None,
            "visemes": result.get("visemes", []) if result else [],
            "rtf": round(syn_ms / audio_ms, 3) if audio_ms else None,
            "error": error,
        })


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
                "loaded":      piper_loaded("atlas"),
                "params":      ["speed"],
            },
            {
                "id":          "jarvis",
                "name":        "JARVIS",
                "character":   "Iron Man AI butler (MCU)",
                "description": "Piper en_GB-alan-medium — warm, articulate British male",
                "sample_rate": 22050,
                "loaded":      piper_loaded("jarvis"),
                "params":      ["speed"],
            },
            {
                "id":          "c3po",
                "name":        "C-3PO",
                "character":   "Star Wars protocol droid",
                "description": "Piper en_GB-alan-medium — formal, precise British male",
                "sample_rate": 22050,
                "loaded":      piper_loaded("jarvis"),
                "params":      ["speed"],
            },
            {
                "id":          "tars",
                "name":        "TARS",
                "character":   "Interstellar mission AI",
                "description": "XTTSv2 (Pyrater/TARS fine-tune) — Bill Irwin voice clone, blunt and dry",
                "sample_rate": 24000,
                "loaded":      xtts_loaded(),
                "params":      ["speed"],
            },
            {
                "id":          "terminator",
                "name":        "TERMINATOR",
                "character":   "T-800 inspired robotic profile",
                "description": "Piper HAL-9000 ONNX profile — cold, mechanical, deliberate",
                "sample_rate": 22050,
                "loaded":      piper_loaded("terminator"),
                "params":      ["speed"],
            },
        ],
        "default": settings.default_voice,
    }


@router.get("/activity", summary="Current state + recent operation log")
async def activity():
    return {
        "state":    _voice_state["current"],
        "since":    _voice_state.get("since"),
        "preview":  _voice_state.get("text_preview"),
        "voice":    _voice_state.get("voice"),
        "log":      list(_voice_log),
    }


@router.get("/health", summary="Service health")
async def health():
    return {
        "status":        "ok",
        "service":       "VoiceService",
        "port":          settings.port,
        "stt_loaded":    stt_svc.is_loaded(),
        "glados_loaded":  glados_loaded(),
        "atlas_loaded":   piper_loaded("atlas"),
        "jarvis_loaded":  piper_loaded("jarvis"),
        "tars_loaded":    xtts_loaded(),
        "terminator_loaded": piper_loaded("terminator"),
        "note":          "Internal utility — called by AIGateway, no auth required.",
    }
