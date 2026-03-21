"""XTTSv2 (Coqui) TTS engine — wraps the Pyrater/TARS fine-tuned model.

Lazy-loads on first synthesis call; keeps speaker conditioning latents
cached in memory so subsequent calls skip the expensive conditioning step.

Requires: TTS (pip install TTS)
Model files: models/xtts/tars/{config.json, model.pth, speakers_xtts.pth, vocab.json, reference.wav}
Sample rate: 24000 Hz (XTTSv2 native output rate)
"""
from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import Optional

from .tts_piper import _clean_tts_text, _letter_visemes

logger = logging.getLogger(__name__)


class XttsTTS:
    """XTTSv2 voice cloning engine. Thread-safe after _load()."""

    sample_rate: int = 24000

    def __init__(self, model_dir: str | Path, reference_wav: str | Path, voice_id: str = "tars"):
        self._model_dir         = Path(model_dir)
        self._reference_wav     = str(reference_wav)
        self._voice_id          = voice_id
        self._model             = None
        self._config            = None
        self._gpt_cond_latent   = None
        self._speaker_embedding = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts

            logger.info("Loading XTTSv2 voice '%s' from %s …", self._voice_id, self._model_dir)
            config = XttsConfig()
            config.load_json(str(self._model_dir / "config.json"))

            model = Xtts.init_from_config(config)
            model.load_checkpoint(config, checkpoint_dir=str(self._model_dir), eval=True)
            model.cpu()

            logger.info("Computing speaker conditioning latents for '%s' …", self._voice_id)
            with torch.no_grad():
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=[self._reference_wav]
                )

            self._config            = config
            self._model             = model
            self._gpt_cond_latent   = gpt_cond_latent
            self._speaker_embedding = speaker_embedding
            logger.info("XTTSv2 '%s' ready.", self._voice_id)
        except Exception as e:
            logger.error("Failed to load XTTSv2 voice '%s': %s", self._voice_id, e)
            raise RuntimeError(f"XTTSv2 voice '{self._voice_id}' unavailable: {e}") from e

    # XTTSv2's GPT context window is ~250 chars; longer inputs are silently truncated.
    _MAX_CHUNK_CHARS = 220

    def _split_chunks(self, text: str) -> list[str]:
        """Split text into sentence-bounded chunks under _MAX_CHUNK_CHARS."""
        import re
        sentences = re.split(r"(?<=[.!?…])\s+", text)
        chunks: list[str] = []
        current = ""
        for s in sentences:
            if not s.strip():
                continue
            candidate = (current + " " + s).strip() if current else s
            if len(candidate) <= self._MAX_CHUNK_CHARS:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single sentence exceeds the limit, hard-split it
                while len(s) > self._MAX_CHUNK_CHARS:
                    chunks.append(s[:self._MAX_CHUNK_CHARS])
                    s = s[self._MAX_CHUNK_CHARS:]
                current = s
        if current:
            chunks.append(current)
        return chunks or [text]

    def _synthesize_chunk(self, text: str, speed: float) -> "np.ndarray":
        import numpy as np
        import torch
        with torch.no_grad():
            out = self._model.inference(
                text,
                "en",
                self._gpt_cond_latent,
                self._speaker_embedding,
                temperature=0.7,
                speed=speed,
            )
        return np.array(out["wav"], dtype=np.float32)

    def synthesize(self, text: str, speed: float = 1.0) -> dict:
        """
        Returns:
            audio       — WAV bytes (PCM 16-bit, 24000 Hz, mono)
            visemes     — list of {viseme_id, offset_ms}
            duration_ms — audio length in ms
        """
        import numpy as np

        self._load()

        clean = _clean_tts_text(text)
        if not clean:
            return {"audio": b"", "visemes": [], "duration_ms": 0}

        chunks = self._split_chunks(clean)
        parts: list[np.ndarray] = []
        for chunk in chunks:
            parts.append(self._synthesize_chunk(chunk, speed))

        samples = np.concatenate(parts) if len(parts) > 1 else parts[0]
        pcm_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_int16.tobytes())

        wav_bytes   = buf.getvalue()
        duration_ms = int(len(pcm_int16) / self.sample_rate * 1000)
        visemes     = _letter_visemes(clean, duration_ms)

        return {
            "audio":       wav_bytes,
            "visemes":     visemes,
            "duration_ms": duration_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_tars_instance: Optional[XttsTTS] = None


def get_tars() -> XttsTTS:
    global _tars_instance
    if _tars_instance is None:
        from ..config import settings
        _tars_instance = XttsTTS(settings.tars_xtts_dir, settings.tars_reference_wav)
    return _tars_instance


def is_loaded() -> bool:
    return _tars_instance is not None and _tars_instance._model is not None
