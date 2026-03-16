"""GLaDOS TTS — Piper ONNX voice model with IPA visemes.

Uses piper-tts with the rokeya71/VITS-Piper-GlaDOS-en-onnx Piper model.
IPA phonemes from Piper synthesis are mapped to the 24-viseme set for lip sync.

Requires: piper-tts, espeak-ng, models/glados/glados.onnx + .onnx.json
"""
import io
import logging
import re
import wave
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# IPA phoneme → 24-viseme ID
# ─────────────────────────────────────────────────────────────────────────────
_IPA_VISEME: dict[str, int] = {
    # Vowels
    "ɑ": 1, "ɐ": 1, "ɒ": 1,          # AA
    "æ": 2,                             # AE
    "ʌ": 3, "ə": 3,                    # AH
    "ɔ": 4,                             # AO
    "aʊ": 5, "a": 5,                   # AW
    "eɪ": 6, "e": 7,                   # AY / EH
    "ɛ": 7,                             # EH
    "ɜ": 8, "ɝ": 8, "ɞ": 8,           # ER
    "ɪ": 9,                             # IH
    "i": 10,                            # IY
    "oʊ": 11, "o": 11,                 # OW
    "ɔɪ": 12,                          # OY
    "ʊ": 13,                            # UH
    "u": 14,                            # UW
    # Consonants
    "b": 15, "p": 15, "m": 15,         # bilabial
    "f": 16, "v": 16,                   # labiodental
    "θ": 17, "ð": 17,                  # dental
    "d": 18, "t": 18, "n": 18,         # alveolar
    "k": 19, "g": 19, "ŋ": 19,        # velar
    "tʃ": 20, "dʒ": 20, "ʃ": 20, "ʒ": 20,  # post-alveolar
    "s": 21, "z": 21,                   # sibilant
    "l": 22, "r": 22, "ɹ": 22,        # lateral / rhotic
    "w": 23, "j": 23,                   # approximant
}

_glados_instance: Optional["GladosTTS"] = None


def is_loaded() -> bool:
    return _glados_instance is not None and _glados_instance._voice is not None


def get_glados() -> "GladosTTS":
    global _glados_instance
    if _glados_instance is None:
        _glados_instance = GladosTTS()
    return _glados_instance


def _wav_duration_ms(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return int(w.getnframes() / w.getframerate() * 1000)


def _ipa_visemes(phonemes: list[str], total_ms: int) -> list[dict]:
    """Map IPA phoneme list to timed viseme events."""
    if not phonemes:
        return [{"viseme_id": 0, "offset_ms": 0}]
    step = total_ms / len(phonemes)
    result = []
    for i, ph in enumerate(phonemes):
        vid = _IPA_VISEME.get(ph, 0)
        result.append({"viseme_id": vid, "offset_ms": int(i * step)})
    return result


class GladosTTS:
    """GLaDOS TTS via Piper.  Call .synthesize(text) -> {"audio": bytes, ...}"""

    sample_rate: int = 22050

    def __init__(self):
        self._voice = None

    def _load(self):
        if self._voice is not None:
            return
        from piper.voice import PiperVoice
        from ..config import settings
        onnx_path = str(settings.glados_onnx)
        json_path = onnx_path + ".json"
        logger.info("Loading Piper GLaDOS model from %s ...", onnx_path)
        self._voice = PiperVoice.load(onnx_path, config_path=json_path, use_cuda=False)
        self.sample_rate = self._voice.config.sample_rate
        logger.info("GLaDOS TTS ready  (sample_rate=%d).", self.sample_rate)

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.333,
        noise_w: float = 0.333,
    ) -> dict:
        self._load()
        clean = re.sub(r"\[[A-Z_]+(?::[^\]]+)?\]", "", text).strip()
        clean = re.sub(r"\s+", " ", clean)
        if not clean:
            return {"audio": b"", "visemes": [], "duration_ms": 0}

        from piper.config import SynthesisConfig
        syn_cfg = SynthesisConfig(
            length_scale=1.0 / max(speed, 0.1),
            noise_scale=noise_scale,
            noise_w_scale=noise_w,
        )

        all_audio = bytearray()
        all_phonemes: list[str] = []

        for chunk in self._voice.synthesize(clean, syn_cfg):
            all_audio.extend(chunk.audio_int16_bytes)
            all_phonemes.extend(chunk.phonemes or [])

        # Wrap raw int16 PCM in a WAV container
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(self.sample_rate)
            wav_out.writeframes(bytes(all_audio))

        wav_bytes = buf.getvalue()
        duration_ms = _wav_duration_ms(wav_bytes)

        return {
            "audio":       wav_bytes,
            "visemes":     _ipa_visemes(all_phonemes, duration_ms),
            "duration_ms": duration_ms,
        }
