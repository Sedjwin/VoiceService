"""Speech-to-Text service using faster-whisper.

Accepts raw WAV bytes (as sent by an ESP32-S3 I2S mic: PCM 16-bit, 16 kHz, mono).
Falls back to treating the bytes as headerless raw PCM if soundfile cannot parse them.
"""
import io
import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)

_model: Optional[object] = None


def is_loaded() -> bool:
    return _model is not None


def get_model():
    """Lazily load the Whisper model (downloads on first use, ~140 MB for base.en)."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        from ..config import settings
        logger.info("Loading Whisper model: %s …", settings.whisper_model)
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("Whisper ready.")
    return _model


def transcribe(wav_bytes: bytes) -> dict:
    """
    Transcribe audio bytes → {"text": str, "duration_ms": int, "language": str}.

    Input: WAV file bytes (preferred) or raw int16 PCM at 16 kHz mono.
    Blocking — wrap in asyncio.to_thread() from async callers.
    """
    t0 = time.monotonic()
    model = get_model()

    # Write to a temp file (faster-whisper accepts file paths)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            vad_filter=True,        # skip silence at the edges
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    finally:
        os.unlink(tmp_path)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return {
        "text": text,
        "language": getattr(info, "language", "en"),
        "duration_ms": elapsed_ms,
    }
