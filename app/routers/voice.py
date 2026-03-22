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
    get_hal, get_k9, get_k9v2, get_jarvis, get_wheatley, get_data,
    is_loaded as piper_loaded,
)

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
    voice: str = Field("glados", description="'glados', 'hal', 'k9', 'k9v2', 'jarvis', 'wheatley', or 'data'")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speaking speed multiplier")
    # VITS expressiveness params (GLaDOS only — ignored for Piper)
    noise_scale: float = Field(0.333, ge=0.0, le=1.0, description="Phoneme variation (expressiveness)")
    noise_w: float     = Field(0.333, ge=0.0, le=1.0, description="Duration variation")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_engine(voice: str) -> GladosTTS | PiperTTS:
    v = voice.lower()
    if v == "hal":      return get_hal()
    if v == "k9":       return get_k9()
    if v == "k9v2":     return get_k9v2()
    if v == "jarvis":   return get_jarvis()
    if v == "wheatley": return get_wheatley()
    if v == "data":     return get_data()
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
            "text": req.text,
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
                "description": "Custom ONNX VITS — trained from Portal game audio",
                "sample_rate": 22050,
                "loaded":      glados_loaded(),
                "params":      ["speed", "noise_scale", "noise_w"],
            },
            {
                "id":          "hal",
                "name":        "HAL 9000",
                "character":   "Sentient computer (2001: A Space Odyssey)",
                "description": "Piper — trained from film audio clips",
                "sample_rate": 22050,
                "loaded":      piper_loaded("hal"),
                "params":      ["speed"],
            },
            {
                "id":          "k9",
                "name":        "K-9",
                "character":   "Robot dog (Doctor Who)",
                "description": "Piper — built specifically for Raspberry Pi K-9 replica",
                "sample_rate": 22050,
                "loaded":      piper_loaded("k9"),
                "params":      ["speed"],
            },
            {
                "id":          "k9v2",
                "name":        "K-9 v2",
                "character":   "Robot dog (Doctor Who) — alternate training",
                "description": "Piper — second training run, slightly different timbre",
                "sample_rate": 22050,
                "loaded":      piper_loaded("k9v2"),
                "params":      ["speed"],
            },
            {
                "id":          "jarvis",
                "name":        "JARVIS",
                "character":   "Tony Stark's AI (Marvel MCU)",
                "description": "Piper — trained to emulate Paul Bettany's JARVIS",
                "sample_rate": 22050,
                "loaded":      piper_loaded("jarvis"),
                "params":      ["speed"],
            },
            {
                "id":          "wheatley",
                "name":        "Wheatley",
                "character":   "Personality core (Portal 2)",
                "description": "Piper — trained from original Portal 2 voice files",
                "sample_rate": 22050,
                "loaded":      piper_loaded("wheatley"),
                "params":      ["speed"],
            },
            {
                "id":          "data",
                "name":        "Commander Data",
                "character":   "Android officer (Star Trek: TNG)",
                "description": "Piper — trained from Star Trek Generations game audio",
                "sample_rate": 22050,
                "loaded":      piper_loaded("data"),
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
        "status":          "ok",
        "service":         "VoiceService",
        "port":            settings.port,
        "stt_loaded":      stt_svc.is_loaded(),
        "glados_loaded":   glados_loaded(),
        "hal_loaded":      piper_loaded("hal"),
        "k9_loaded":       piper_loaded("k9"),
        "k9v2_loaded":     piper_loaded("k9v2"),
        "jarvis_loaded":   piper_loaded("jarvis"),
        "wheatley_loaded": piper_loaded("wheatley"),
        "data_loaded":     piper_loaded("data"),
        "note":            "Internal utility — called by AIGateway, no auth required.",
    }
