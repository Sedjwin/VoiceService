"""ATLAS TTS service — Piper TTS with en_US-ryan-high voice.

ATLAS is the Portal 2 robot companion — clear, professional, AI-assistant tone.
Uses the piper-tts Python package which wraps the Piper ONNX VITS voice models.

Requires: piper-tts (pip), en_US-ryan-high.onnx + .json (download_models.py)
Sample rate: 22050 Hz
"""
import io
import logging
import re
import wave
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Approximate letter → viseme mapping for Piper
# (Piper doesn't expose per-phoneme timing, so we estimate from text)
# ─────────────────────────────────────────────────────────────────────────────
_LETTER_VISEME: dict[str, int] = {
    "a": 1, "e": 4, "i": 6, "o": 7, "u": 8,
    "b": 11, "p": 11, "m": 11,
    "f": 12, "v": 12,
    "t": 14, "d": 14,
    "s": 15, "z": 15,
    "n": 17, "g": 20, "k": 20,
    "l": 18, "r": 19, "w": 21, "h": 22, "y": 23,
    " ": 0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────
def _clean_tts_text(text: str) -> str:
    """Strip markdown/symbols that espeak reads aloud literally (asterisk, hash, etc.)."""
    # Bold / italic — extract inner content
    text = re.sub(r'\*{1,3}([^*\n]*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}([^_\n]*?)_{1,2}', r'\1', text)
    # Strikethrough
    text = re.sub(r'~~([^~]*?)~~', r'\1', text)
    # Headers — strip # markers, keep text
    text = re.sub(r'(?m)^#{1,6}\s+', '', text)
    # Fenced code blocks — drop entirely
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Inline code — keep content
    text = re.sub(r'`([^`\n]*?)`', r'\1', text)
    # Block quotes
    text = re.sub(r'(?m)^>\s?', '', text)
    # Markdown links / images → text
    text = re.sub(r'!?\[([^\]]*?)\]\([^)]*?\)', r'\1', text)
    # Remaining action tags (already stripped upstream, belt-and-braces)
    text = re.sub(r'\[[^\]]*?\]', '', text)
    # Table pipes
    text = re.sub(r'\|', ' ', text)
    # Horizontal rules
    text = re.sub(r'(?m)^[-*_]{3,}\s*$', '', text)
    # Remaining symbols espeak reads literally
    text = re.sub(r'[*_\\^~`<>{#]', '', text)
    # Multiple newlines → spoken pause
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


_atlas_instance: Optional["AtlasTTS"] = None


def is_loaded() -> bool:
    return _atlas_instance is not None and _atlas_instance._voice is not None


def get_atlas() -> "AtlasTTS":
    global _atlas_instance
    if _atlas_instance is None:
        _atlas_instance = AtlasTTS()
    return _atlas_instance


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wav_duration_ms(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return int(w.getnframes() / w.getframerate() * 1000)


def _letter_visemes(text: str, total_ms: int) -> list[dict]:
    letters = [c.lower() for c in text if c.isalpha() or c == " "]
    if not letters:
        return [{"viseme_id": 0, "offset_ms": 0}]
    step = total_ms / len(letters)
    return [
        {"viseme_id": _LETTER_VISEME.get(c, 0), "offset_ms": int(i * step)}
        for i, c in enumerate(letters)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class AtlasTTS:
    """ATLAS TTS via Piper.  Call .synthesize(text) → {"audio": bytes, ...}"""

    sample_rate: int = 22050

    def __init__(self):
        self._voice = None

    def _load(self):
        if self._voice is not None:
            return
        try:
            from piper.voice import PiperVoice
            from ..config import settings

            onnx_path = str(settings.piper_voice_onnx)
            json_path = onnx_path + ".json"

            logger.info("Loading Piper ATLAS model from %s …", onnx_path)
            self._voice = PiperVoice.load(onnx_path, config_path=json_path, use_cuda=False)
            self.sample_rate = self._voice.config.sample_rate
            logger.info("ATLAS TTS ready  (sample_rate=%d).", self.sample_rate)
        except Exception as e:
            logger.error("Failed to load Piper ATLAS: %s", e)
            raise RuntimeError(f"ATLAS TTS unavailable: {e}") from e

    def synthesize(self, text: str, speed: float = 1.0) -> dict:
        """
        Returns:
            audio       — WAV bytes (PCM 16-bit, sample_rate Hz, mono)
            visemes     — list of {viseme_id, offset_ms}
            duration_ms — audio length in ms
        Blocking — wrap in asyncio.to_thread() from async callers.
        """
        self._load()

        clean = _clean_tts_text(text)
        if not clean:
            return {"audio": b"", "visemes": [], "duration_ms": 0}

        from piper.config import SynthesisConfig
        syn_cfg = SynthesisConfig(
            length_scale=1.0 / max(speed, 0.1),
            noise_scale=0.667,
            noise_w_scale=0.8,
        )

        all_audio: bytearray = bytearray()
        for chunk in self._voice.synthesize(clean, syn_cfg):
            all_audio.extend(chunk.audio_int16_bytes)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(self.sample_rate)
            wav_out.writeframes(bytes(all_audio))

        wav_bytes   = buf.getvalue()
        duration_ms = _wav_duration_ms(wav_bytes)
        visemes     = _letter_visemes(clean, duration_ms)

        return {
            "audio":      wav_bytes,
            "visemes":    visemes,
            "duration_ms": duration_ms,
        }
