"""Voice API router.

All endpoints are ESP32-friendly:
  - POST /stt          raw WAV bytes in body → JSON transcript
  - POST /tts          JSON request → JSON with base64 audio + visemes
  - POST /tts/raw      JSON request → raw WAV bytes
  - POST /voice/chat   multipart (audio file + form fields) → full pipeline JSON
  - GET  /voices       list voices + loaded status
  - GET  /health       service health
"""
import asyncio
import base64
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..config import settings
from ..services import stt as stt_svc
from ..services.pipeline import BUFFER_BYTES, parse_actions, voice_chat
from ..services.tts_glados import GladosTTS, get_glados, is_loaded as glados_loaded
from ..services.tts_piper import AtlasTTS, get_atlas, is_loaded as atlas_loaded

router = APIRouter(tags=["Voice"])
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: str = Field("glados", description="'glados' or 'atlas'")
    speed: float = Field(1.0, ge=0.25, le=4.0)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_engine(voice: str) -> GladosTTS | AtlasTTS:
    if voice.lower() == "atlas":
        return get_atlas()
    return get_glados()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/stt", summary="Speech → Text (WAV bytes in body)")
async def stt_endpoint(request: Request):
    """
    Send raw WAV bytes (PCM 16-bit 16 kHz mono) in the HTTP body.

    ESP32 example:
        client.println("POST /stt HTTP/1.1");
        client.println("Content-Type: audio/wav");
        client.println("Content-Length: " + String(len));
        client.write(buf, len);
    """
    wav_bytes = await request.body()
    if not wav_bytes:
        raise HTTPException(400, "Empty body — send raw WAV bytes.")
    result = await asyncio.to_thread(stt_svc.transcribe, wav_bytes)
    return result


@router.post("/tts", summary="Text → Audio (JSON, base64 response)")
async def tts_json_endpoint(req: TTSRequest):
    """
    Returns JSON with base64-encoded WAV audio, visemes, and buffer hint.
    Use `buffer_bytes` to know how much audio to pre-load before playback.
    """
    engine = _get_engine(req.voice)
    result = await asyncio.to_thread(engine.synthesize, req.text, req.speed)
    return {
        "voice":       req.voice,
        "audio":       base64.b64encode(result["audio"]).decode(),
        "audio_format":"wav",
        "sample_rate": engine.sample_rate,
        "duration_ms": result["duration_ms"],
        "buffer_bytes": BUFFER_BYTES,
        "visemes":     result["visemes"],
    }


@router.post("/tts/raw", summary="Text → Audio (raw WAV bytes)")
async def tts_raw_endpoint(req: TTSRequest):
    """Returns the WAV file directly — useful for browser/curl testing."""
    engine = _get_engine(req.voice)
    result = await asyncio.to_thread(engine.synthesize, req.text, req.speed)
    return Response(
        content=result["audio"],
        media_type="audio/wav",
        headers={"X-Duration-Ms": str(result["duration_ms"])},
    )


@router.post("/voice/chat", summary="Full pipeline: audio → STT → LLM → TTS")
async def voice_chat_endpoint(
    audio: UploadFile = File(..., description="WAV audio from ESP32 mic"),
    voice: str               = Form("glados",   description="'glados' or 'atlas'"),
    speed: float             = Form(1.0),
    model: str               = Form("",         description="AIGateway model ID (empty = auto)"),
    api_key: str             = Form("",         description="AIGateway agent Bearer token"),
    system_prompt: str       = Form("",         description="Override default LLM system prompt"),
    max_tokens: int          = Form(0),
    temperature: float       = Form(0.0),
):
    """
    Complete voice interaction loop:
    1. Transcribe audio (Whisper STT)
    2. Send transcript to AIGateway for LLM response
    3. Parse action tags  ([HAPPY], [ANGRY], [COLOR:red] etc.)
    4. Synthesise speech (GLaDOS or ATLAS TTS)
    5. Return JSON with audio, visemes, actions

    Supports ESP32 multipart POST — audio field is a WAV file upload.
    """
    wav_bytes = await audio.read()
    if not wav_bytes:
        raise HTTPException(400, "No audio data received.")

    config = {
        "voice":         voice,
        "speed":         speed,
        "model":         model,
        "api_key":       api_key or settings.aigateway_api_key,
        "system_prompt": system_prompt,
        "max_tokens":    max_tokens,
        "temperature":   temperature,
    }
    return await voice_chat(wav_bytes, config)


@router.post("/voice/tts-chat", summary="Text → LLM → TTS (no STT step)")
async def text_chat_endpoint(
    text: str         = Form(...),
    voice: str        = Form("glados"),
    speed: float      = Form(1.0),
    model: str        = Form(""),
    api_key: str      = Form(""),
    system_prompt: str= Form(""),
):
    """
    Like /voice/chat but accepts text directly — skips the STT step.
    Useful for testing LLM + TTS without a microphone.
    """
    from ..services.pipeline import _call_llm

    config = {
        "model":         model,
        "api_key":       api_key or settings.aigateway_api_key,
        "system_prompt": system_prompt,
    }
    llm_result  = await _call_llm(text, config)
    response_text = llm_result["content"]
    clean_text, actions = parse_actions(response_text)

    engine = _get_engine(voice)
    tts_result = await asyncio.to_thread(engine.synthesize, clean_text, speed)

    return {
        "input_text":    text,
        "response_text": response_text,
        "clean_text":    clean_text,
        "actions":       actions,
        "voice":         voice,
        "model_used":    llm_result["model"],
        "audio":         base64.b64encode(tts_result["audio"]).decode(),
        "audio_format":  "wav",
        "sample_rate":   engine.sample_rate,
        "duration_ms":   tts_result["duration_ms"],
        "buffer_bytes":  BUFFER_BYTES,
        "visemes":       tts_result["visemes"],
    }


@router.get("/voices", summary="List available TTS voices")
async def list_voices():
    return {
        "voices": [
            {
                "id":          "glados",
                "name":        "GLaDOS",
                "character":   "Aperture Science AI  (Portal)",
                "description": "Precise, condescending, darkly humorous ONNX VITS voice",
                "sample_rate": 22050,
                "loaded":      glados_loaded(),
            },
            {
                "id":          "atlas",
                "name":        "ATLAS",
                "character":   "Cooperative android  (Portal 2)",
                "description": "Clear, professional AI-assistant voice (Piper en_US-ryan-high)",
                "sample_rate": 22050,
                "loaded":      atlas_loaded(),
            },
        ],
        "default": settings.default_voice,
    }


@router.get("/health", summary="Service health")
async def health():
    return {
        "status":       "ok",
        "service":      "VoiceService",
        "port":         settings.port,
        "aigateway":    settings.aigateway_url,
        "stt_loaded":   stt_svc.is_loaded(),
        "glados_loaded": glados_loaded(),
        "atlas_loaded": atlas_loaded(),
    }
