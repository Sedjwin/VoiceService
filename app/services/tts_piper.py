"""Generic Piper TTS engine — wraps ONNX VITS piper-tts for any voice model.

Currently used for:
  ATLAS  — en_US-ryan-high (professional male)
  JARVIS — en_GB-alan-medium (warm British male)
  TARS   — en_US-hfc_male-medium (direct, precise US male)

Requires: piper-tts (pip), matching .onnx + .json model files.
Sample rate: 22050 Hz (all piper-voices models).
"""
import io
import logging
import re
import wave
from pathlib import Path
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


def _clean_tts_text(text: str) -> str:
    """Strip markdown/symbols that espeak reads aloud literally."""
    text = re.sub(r'\*{1,3}([^*\n]*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}([^_\n]*?)_{1,2}', r'\1', text)
    text = re.sub(r'~~([^~]*?)~~', r'\1', text)
    text = re.sub(r'(?m)^#{1,6}\s+', '', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`\n]*?)`', r'\1', text)
    text = re.sub(r'(?m)^>\s?', '', text)
    text = re.sub(r'!?\[([^\]]*?)\]\([^)]*?\)', r'\1', text)
    text = re.sub(r'\[[^\]]*?\]', '', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'(?m)^[-*_]{3,}\s*$', '', text)
    text = re.sub(r'[*_\\^~`<>{#]', '', text)
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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
# Generic engine
# ─────────────────────────────────────────────────────────────────────────────

class PiperTTS:
    """Generic Piper TTS. Instantiate with a model path; call .synthesize(text)."""

    sample_rate: int = 22050

    def __init__(self, model_path: str | Path, voice_id: str = "piper"):
        self._model_path = str(model_path)
        self._voice_id   = voice_id
        self._voice      = None

    def _load(self) -> None:
        if self._voice is not None:
            return
        try:
            from piper.voice import PiperVoice
            json_path = self._model_path + ".json"
            logger.info("Loading Piper voice '%s' from %s …", self._voice_id, self._model_path)
            self._voice = PiperVoice.load(self._model_path, config_path=json_path, use_cuda=False)
            self.sample_rate = self._voice.config.sample_rate
            logger.info("Piper '%s' ready (sample_rate=%d).", self._voice_id, self.sample_rate)
        except Exception as e:
            logger.error("Failed to load Piper voice '%s': %s", self._voice_id, e)
            raise RuntimeError(f"Piper voice '{self._voice_id}' unavailable: {e}") from e

    def synthesize(self, text: str, speed: float = 1.0) -> dict:
        """
        Returns:
            audio       — WAV bytes (PCM 16-bit, sample_rate Hz, mono)
            visemes     — list of {viseme_id, offset_ms}
            duration_ms — audio length in ms
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
            "audio":       wav_bytes,
            "visemes":     visemes,
            "duration_ms": duration_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-voice singletons
# ─────────────────────────────────────────────────────────────────────────────

_atlas_instance:  Optional[PiperTTS] = None
_jarvis_instance: Optional[PiperTTS] = None
_tars_instance:   Optional[PiperTTS] = None


def get_atlas() -> PiperTTS:
    global _atlas_instance
    if _atlas_instance is None:
        from ..config import settings
        _atlas_instance = PiperTTS(settings.piper_voice_onnx, "atlas")
    return _atlas_instance


def get_jarvis() -> PiperTTS:
    global _jarvis_instance
    if _jarvis_instance is None:
        from ..config import settings
        _jarvis_instance = PiperTTS(settings.jarvis_voice_onnx, "jarvis")
    return _jarvis_instance


def get_tars() -> PiperTTS:
    global _tars_instance
    if _tars_instance is None:
        from ..config import settings
        _tars_instance = PiperTTS(settings.tars_voice_onnx, "tars")
    return _tars_instance


def is_loaded(voice_id: str = "atlas") -> bool:
    inst = {"atlas": _atlas_instance, "jarvis": _jarvis_instance, "tars": _tars_instance}.get(voice_id)
    return inst is not None and inst._voice is not None


# Backward-compat alias
AtlasTTS = PiperTTS
