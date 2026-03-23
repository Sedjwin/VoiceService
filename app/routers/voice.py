"""Voice API router — pure STT and TTS endpoints.

VoiceService is an internal utility called BY AIGateway.
It has no auth, no LLM calls, and no agent awareness.
AIGateway owns all orchestration and agent authentication.

Endpoints:
  POST /stt                              raw WAV bytes → transcript JSON
  POST /tts                              {text, voice, speed?, ...} → base64 WAV + visemes
  POST /tts/raw                          {text, voice, speed?}      → raw WAV bytes
  GET  /voices                           list available voices
  GET  /voices/{voice_id}/settings       per-voice tuning params
  PUT  /voices/{voice_id}/settings       update per-voice tuning params (persisted to disk)
  GET  /models                           installed models + load status
  POST /models/{voice_id}/load           force-load a model into memory
  POST /models/{voice_id}/unload         free a model from memory
  POST /models/{voice_id}/interrupt      block next synthesis (returns 503)
  GET  /stats                            CPU, RAM, temperature
  GET  /activity                         current state + recent operation log
  GET  /health                           service status
"""
import asyncio
import base64
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import psutil
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

# ── Model management ─────────────────────────────────────────────────────────
_blocked_voices: set[str] = set()   # voices whose next synthesis returns 503

# ── Voice metadata registry ──────────────────────────────────────────────────
_VOICE_META = [
    {"id": "glados",   "name": "GLaDOS",         "character": "Aperture Science AI (Portal)"},
    {"id": "hal",      "name": "HAL 9000",        "character": "Sentient computer (2001: A Space Odyssey)"},
    {"id": "k9",       "name": "K-9",             "character": "Robot dog (Doctor Who)"},
    {"id": "k9v2",     "name": "K-9 v2",          "character": "Robot dog — alternate training"},
    {"id": "jarvis",   "name": "JARVIS",           "character": "Tony Stark's AI (Marvel MCU)"},
    {"id": "wheatley", "name": "Wheatley",         "character": "Personality core (Portal 2)"},
    {"id": "data",     "name": "Commander Data",   "character": "Android officer (Star Trek: TNG)"},
]
_VOICE_IDS = {m["id"] for m in _VOICE_META}


def _voice_file_path(voice_id: str) -> Path:
    return Path(str({
        "glados":   settings.glados_onnx,
        "hal":      settings.hal_onnx,
        "k9":       settings.k9_onnx,
        "k9v2":     settings.k9v2_onnx,
        "jarvis":   settings.jarvis_onnx,
        "wheatley": settings.wheatley_onnx,
        "data":     settings.data_onnx,
    }.get(voice_id, "/nonexistent")))


# ── Per-voice tuning settings (persisted to data/voice_settings.json) ────────
_SETTINGS_FILE = Path(__file__).parent.parent.parent / "data" / "voice_settings.json"

# All models are character-trained — 1.0 is correct for every voice.
# The only known exception is Wheatley whose model runs faster than the character's
# actual delivery; adjust via the Tune panel if needed.
_DEFAULT_SETTINGS: dict[str, dict] = {
    "glados":   {"speed": 1.0, "noise_scale": 0.333, "noise_w": 0.333},
    "hal":      {"speed": 1.0},
    "k9":       {"speed": 1.0},
    "k9v2":     {"speed": 1.0},
    "jarvis":   {"speed": 1.0},
    "wheatley": {"speed": 1.0},
    "data":     {"speed": 1.0},
}


def _load_voice_settings() -> dict:
    try:
        if _SETTINGS_FILE.exists():
            saved = json.loads(_SETTINGS_FILE.read_text())
            # Merge: ensure all voices present, defaults fill missing keys
            merged = {k: dict(v) for k, v in _DEFAULT_SETTINGS.items()}
            for vid, vals in saved.items():
                if vid in merged:
                    merged[vid].update(vals)
            return merged
    except Exception as e:
        logger.warning("Could not load voice_settings.json: %s — using defaults", e)
    return {k: dict(v) for k, v in _DEFAULT_SETTINGS.items()}


def _save_voice_settings() -> None:
    try:
        _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_FILE.write_text(json.dumps(_voice_settings, indent=2))
    except Exception as e:
        logger.error("Could not save voice_settings.json: %s", e)


_voice_settings: dict = _load_voice_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: str = Field("glados", description="Voice ID")
    # All synthesis params are optional — if None, the per-voice stored setting is used
    speed:       Optional[float] = Field(None, ge=0.25, le=4.0,  description="Speed override. Omit to use stored per-voice setting.")
    noise_scale: Optional[float] = Field(None, ge=0.0,  le=1.0,  description="GLaDOS: phoneme variation. Omit to use stored setting.")
    noise_w:     Optional[float] = Field(None, ge=0.0,  le=1.0,  description="GLaDOS: duration variation. Omit to use stored setting.")


class VoiceSettingsUpdate(BaseModel):
    speed:       float          = Field(1.0,  ge=0.25, le=4.0)
    noise_scale: Optional[float] = Field(None, ge=0.0,  le=1.0)   # GLaDOS only
    noise_w:     Optional[float] = Field(None, ge=0.0,  le=1.0)   # GLaDOS only


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


def _resolve_tts_kwargs(voice_id: str, req: TTSRequest) -> dict:
    """Merge stored per-voice settings with any explicit request overrides."""
    stored = _voice_settings.get(voice_id, {})
    speed  = req.speed if req.speed is not None else stored.get("speed", 1.0)
    kwargs: dict = {"speed": speed}
    if voice_id == "glados":
        kwargs["noise_scale"] = req.noise_scale if req.noise_scale is not None else stored.get("noise_scale", 0.333)
        kwargs["noise_w"]     = req.noise_w     if req.noise_w     is not None else stored.get("noise_w",     0.333)
    return kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — STT / TTS
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
    voice_id = req.voice.lower()

    if voice_id in _blocked_voices:
        _blocked_voices.discard(voice_id)
        raise HTTPException(503, f"Voice '{req.voice}' synthesis was interrupted by admin.")

    engine = _get_engine(req.voice)
    kwargs = _resolve_tts_kwargs(voice_id, req)

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
            "speed": kwargs.get("speed"),
            "noise_scale": kwargs.get("noise_scale"),
            "noise_w":     kwargs.get("noise_w"),
            "audio_ms":    audio_ms,
            "viseme_count": len(result.get("visemes", [])) if result else None,
            "visemes": result.get("visemes", []) if result else [],
            "rtf": round(syn_ms / audio_ms, 3) if audio_ms else None,
            "error": error,
        })


@router.post("/tts/raw", summary="Text → audio (raw WAV bytes)")
async def tts_raw_endpoint(req: TTSRequest):
    """Returns WAV directly — useful for curl testing or non-ESP32 clients."""
    voice_id = req.voice.lower()
    if voice_id in _blocked_voices:
        _blocked_voices.discard(voice_id)
        raise HTTPException(503, f"Voice '{req.voice}' synthesis was interrupted by admin.")
    engine = _get_engine(req.voice)
    kwargs = _resolve_tts_kwargs(voice_id, req)
    result = await asyncio.to_thread(engine.synthesize, req.text, **kwargs)
    return Response(
        content=result["audio"],
        media_type="audio/wav",
        headers={"X-Duration-Ms": str(result["duration_ms"])},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — Voice settings (per-voice tuning, persisted)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/voices/{voice_id}/settings", summary="Get per-voice tuning parameters")
async def get_voice_settings(voice_id: str):
    if voice_id not in _VOICE_IDS:
        raise HTTPException(404, f"Unknown voice: '{voice_id}'")
    stored = _voice_settings.get(voice_id, {})
    defaults = _DEFAULT_SETTINGS.get(voice_id, {})
    return {
        "voice_id":    voice_id,
        "speed":       stored.get("speed",       defaults.get("speed", 1.0)),
        "noise_scale": stored.get("noise_scale", defaults.get("noise_scale")),
        "noise_w":     stored.get("noise_w",     defaults.get("noise_w")),
        "defaults":    defaults,
    }


@router.put("/voices/{voice_id}/settings", summary="Update per-voice tuning parameters (persisted)")
async def update_voice_settings(voice_id: str, req: VoiceSettingsUpdate):
    if voice_id not in _VOICE_IDS:
        raise HTTPException(404, f"Unknown voice: '{voice_id}'")

    new: dict = {"speed": req.speed}
    if voice_id == "glados":
        cur = _voice_settings.get("glados", {})
        new["noise_scale"] = req.noise_scale if req.noise_scale is not None else cur.get("noise_scale", 0.333)
        new["noise_w"]     = req.noise_w     if req.noise_w     is not None else cur.get("noise_w",     0.333)

    _voice_settings[voice_id] = new
    _save_voice_settings()
    logger.info("Voice settings updated: %s → %s", voice_id, new)
    return {"voice_id": voice_id, **new}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — Model management
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/models", summary="List all models with install and load status")
async def list_models():
    current_voice = _voice_state.get("voice") if _voice_state["current"] == "tts" else None

    tts_models = []
    for m in _VOICE_META:
        vid = m["id"]
        fp  = _voice_file_path(vid)
        exists = fp.exists()
        loaded = glados_loaded() if vid == "glados" else piper_loaded(vid)
        stored = _voice_settings.get(vid, {})
        tts_models.append({
            "id":           vid,
            "name":         m["name"],
            "character":    m["character"],
            "type":         "tts",
            "file_exists":  exists,
            "file_size_mb": round(fp.stat().st_size / 1024 / 1024, 1) if exists else None,
            "loaded":       loaded,
            "active":       vid == current_voice,
            "blocked":      vid in _blocked_voices,
            "speed":        stored.get("speed", 1.0),
        })

    stt_entry = {
        "id":           "stt",
        "name":         "Whisper STT",
        "character":    f"faster-whisper · {settings.whisper_model}",
        "type":         "stt",
        "file_exists":  True,
        "file_size_mb": None,
        "loaded":       stt_svc.is_loaded(),
        "active":       _voice_state["current"] == "stt",
        "blocked":      False,
        "speed":        None,
    }

    return {"models": [stt_entry] + tts_models}


@router.post("/models/{voice_id}/load", summary="Force-load a model into memory")
async def load_model(voice_id: str):
    _blocked_voices.discard(voice_id)
    if voice_id == "stt":
        await asyncio.to_thread(stt_svc.get_model)
        return {"voice_id": "stt", "loaded": True}
    if voice_id not in _VOICE_IDS:
        raise HTTPException(404, f"Unknown voice: '{voice_id}'")
    engine = _get_engine(voice_id)
    await asyncio.to_thread(engine._load)
    return {"voice_id": voice_id, "loaded": True}


@router.post("/models/{voice_id}/unload", summary="Unload a model from memory")
async def unload_model(voice_id: str):
    if voice_id == "stt":
        stt_svc._model = None
        return {"voice_id": "stt", "loaded": False}
    if voice_id not in _VOICE_IDS:
        raise HTTPException(404, f"Unknown voice: '{voice_id}'")
    engine = _get_engine(voice_id)
    engine._voice = None
    return {"voice_id": voice_id, "loaded": False}


@router.post("/models/{voice_id}/interrupt", summary="Block next synthesis for this voice (returns 503)")
async def interrupt_model(voice_id: str):
    if voice_id not in _VOICE_IDS:
        raise HTTPException(404, f"Unknown voice: '{voice_id}'")
    _blocked_voices.add(voice_id)
    return {"voice_id": voice_id, "blocked": True, "message": "Next synthesis request will return 503."}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — System stats
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stats", summary="System CPU, RAM, temperature")
async def system_stats():
    def _collect():
        cpu = psutil.cpu_percent(interval=0.3)
        mem = psutil.virtual_memory()
        temp = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name in ("cpu_thermal", "rp1_adc", "bcm2835_thermal", "coretemp"):
                    if name in temps and temps[name]:
                        temp = round(temps[name][0].current, 1)
                        break
                if temp is None:
                    first = next(iter(temps.values()), None)
                    if first:
                        temp = round(first[0].current, 1)
        except Exception:
            pass
        return {
            "cpu_percent":     round(cpu, 1),
            "memory_used_mb":  round(mem.used  / 1024 / 1024, 0),
            "memory_total_mb": round(mem.total / 1024 / 1024, 0),
            "memory_percent":  round(mem.percent, 1),
            "temperature_c":   temp,
        }
    return await asyncio.to_thread(_collect)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — Info
# ─────────────────────────────────────────────────────────────────────────────

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
