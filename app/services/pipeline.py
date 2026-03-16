"""Full voice pipeline: audio → STT → AIGateway LLM → TTS → response.

Response contract (what the ESP32-S3 receives):
{
    "input_text":    "transcribed user speech",
    "response_text": "raw LLM output (may include action tags)",
    "clean_text":    "spoken text (action tags removed)",
    "actions":       [{"type": "expression", "value": "happy"}, ...],
    "voice":         "glados" | "atlas",
    "audio":         "<base64 WAV>",
    "audio_format":  "wav",
    "sample_rate":   22050,
    "duration_ms":   1400,
    "buffer_bytes":  22050,   # pre-buffer this many bytes before playback
    "pipeline_ms":   850,     # total wall-clock time
}

Action tag format in LLM response:
  [HAPPY]  [ANGRY]  [SAD]  [THINKING]  [SURPRISED]  [NEUTRAL]
  [NOD]    [SHAKE]  [BLINK]
  [COLOR:red]  [COLOR:blue]  [COLOR:green]
"""
import asyncio
import base64
import logging
import re
import time
from typing import Any

import httpx

from ..config import settings
from . import stt as stt_svc
from .tts_glados import get_glados
from .tts_piper import get_atlas

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Buffer hint: bytes to pre-load before ESP32 starts I2S playback
# = 0.5 s × 22050 Hz × 1 channel × 2 bytes/sample
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050
BUFFER_BYTES  = int(SAMPLE_RATE * settings.buffer_hint_ms / 1000 * 2)

# ─────────────────────────────────────────────────────────────────────────────
# Action tag parsing
# ─────────────────────────────────────────────────────────────────────────────
_ACTION_RE = re.compile(r"\[([A-Z_]+(?::[a-z0-9]+)?)\]")


def parse_actions(text: str) -> tuple[str, list[dict]]:
    """Strip action tags from text, return (clean_text, [action_dicts])."""
    actions: list[dict] = []
    for match in _ACTION_RE.finditer(text):
        tag = match.group(1)
        if ":" in tag:
            action_type, value = tag.split(":", 1)
            actions.append({"type": action_type.lower(), "value": value})
        else:
            actions.append({"type": "expression", "value": tag.lower()})

    clean = _ACTION_RE.sub("", text).strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean, actions


# ─────────────────────────────────────────────────────────────────────────────
# LLM call via AIGateway
# ─────────────────────────────────────────────────────────────────────────────

async def _call_llm(user_text: str, config: dict[str, Any]) -> dict:
    system_prompt = config.get("system_prompt") or settings.llm_system_prompt
    model         = config.get("model") or settings.llm_model
    max_tokens    = int(config.get("max_tokens") or settings.llm_max_tokens)
    temperature   = float(config.get("temperature") or settings.llm_temperature)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_text},
    ]
    payload = {
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }
    if model:
        payload["model"] = model

    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = config.get("api_key") or settings.aigateway_api_key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
        resp = await client.post(
            f"{settings.aigateway_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    return {
        "content": content,
        "model": data.get("model", "unknown"),
        "usage": data.get("usage", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def voice_chat(wav_bytes: bytes, config: dict[str, Any]) -> dict:
    """
    Full voice pipeline.

    Args:
        wav_bytes: Raw WAV audio from ESP32 (PCM 16-bit 16 kHz mono preferred).
        config:    Per-request overrides — see .env.example for key names.

    Returns:
        ESP32 response dict (see module docstring).
    """
    t0 = time.monotonic()

    # ── 1. STT ────────────────────────────────────────────────────────────────
    stt_result  = await asyncio.to_thread(stt_svc.transcribe, wav_bytes)
    user_text   = stt_result["text"]
    if not user_text:
        return {"error": "Could not transcribe audio — empty transcript.", "input_text": ""}

    logger.info("STT: %r", user_text)

    # ── 2. LLM via AIGateway ──────────────────────────────────────────────────
    llm_result    = await _call_llm(user_text, config)
    response_text = llm_result["content"]
    logger.info("LLM: %r", response_text[:120])

    # ── 3. Parse action tags ──────────────────────────────────────────────────
    clean_text, actions = parse_actions(response_text)

    # ── 4. TTS ────────────────────────────────────────────────────────────────
    voice = (config.get("voice") or settings.default_voice).lower()
    speed = float(config.get("speed") or 1.0)

    if voice == "atlas":
        tts_engine = get_atlas()
    else:
        tts_engine = get_glados()
        voice = "glados"

    tts_result = await asyncio.to_thread(tts_engine.synthesize, clean_text, speed)

    # ── 5. Assemble response ──────────────────────────────────────────────────
    pipeline_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Voice pipeline done in %d ms  (STT+%d ms TTS)",
        pipeline_ms, tts_result["duration_ms"],
    )

    return {
        "input_text":    user_text,
        "response_text": response_text,
        "clean_text":    clean_text,
        "actions":       actions,
        "voice":         voice,
        "model_used":    llm_result["model"],
        "audio":         base64.b64encode(tts_result["audio"]).decode(),
        "audio_format":  "wav",
        "sample_rate":   tts_engine.sample_rate,
        "duration_ms":   tts_result["duration_ms"],
        "buffer_bytes":  BUFFER_BYTES,
        "visemes":       tts_result["visemes"],
        "pipeline_ms":   pipeline_ms,
    }
