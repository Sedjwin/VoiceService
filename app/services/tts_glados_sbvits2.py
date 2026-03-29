"""GLaDOS TTS — Style-Bert-VITS2 (WarriorMama777/GLaDOS_TTS).

High-quality BERT-enhanced VITS2 model trained on Portal 1 & 2 voice lines.
Uses microsoft/deberta-v3-large for prosody conditioning.

Five speaking styles available:
  Neutral, Standard, Deep, Light, Standard_02

Sample rate: 44100 Hz (higher quality than the ONNX Piper/dnhkng voices at 22050 Hz)

Requires: style-bert-vits2, torch, transformers
Models: models/sbvits2_glados/ + models/bert/deberta-v3-large/
"""
from __future__ import annotations

import io
import logging
import re
import wave
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR   = Path(__file__).parent.parent.parent / "models" / "sbvits2_glados"
_BERT_DIR    = Path(__file__).parent.parent.parent / "models" / "bert" / "deberta-v3-large"

STYLES = ["Neutral", "Standard", "Deep", "Light", "Standard_02"]


def _clean_tts_text(text: str) -> str:
    text = text.replace("…", "...")
    text = re.sub(r"\.{3,}", ". ", text)
    text = re.sub(r"\s*[—–]{2,}\s*", ". ", text)
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    text = re.sub(r"\*{1,3}([^*\n]*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_\n]*?)_{1,2}", r"\1", text)
    text = re.sub(r"~~([^~]*?)~~", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`\n]*?)`", r"\1", text)
    text = re.sub(r"(?m)^>\s?", "", text)
    text = re.sub(r"!?\[([^\]]*?)\]\([^)]*?\)", r"\1", text)
    text = re.sub(r"\[[^\]]*?\]", "", text)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"(?m)^[-*_]{3,}\s*$", "", text)
    text = re.sub(r"[*_\\^~`<>{#]", "", text)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _letter_visemes(text: str, total_ms: int) -> list[dict]:
    """Letter-based viseme approximation (same as piper voices)."""
    _MAP = {
        "a": 1, "e": 7, "i": 10, "o": 11, "u": 14,
        "b": 15, "m": 15, "p": 15,
        "f": 16, "v": 16,
        "t": 18, "d": 18, "n": 18, "l": 22,
        "s": 21, "z": 21,
        "k": 19, "g": 19,
        "r": 22, "w": 23, "h": 0, "y": 23,
    }
    chars = [c for c in text.lower() if c.isalpha()]
    if not chars:
        return [{"viseme_id": 0, "offset_ms": 0}]
    step = total_ms / len(chars)
    return [{"viseme_id": _MAP.get(c, 0), "offset_ms": int(i * step)} for i, c in enumerate(chars)]


# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional["GladosSbVits2TTS"] = None


def is_loaded() -> bool:
    return _instance is not None and _instance._loaded


def get_glados_sbvits2() -> "GladosSbVits2TTS":
    global _instance
    if _instance is None:
        _instance = GladosSbVits2TTS()
    return _instance


class GladosSbVits2TTS:
    """GLaDOS TTS via Style-Bert-VITS2 (WarriorMama777).

    Call .synthesize(text) → {"audio": bytes, "visemes": [...], "duration_ms": int}
    """

    sample_rate: int = 44100

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not (_MODEL_DIR / "model.safetensors").exists():
            raise FileNotFoundError(f"Style-Bert-VITS2 GLaDOS model not found: {_MODEL_DIR}")
        if not (_BERT_DIR / "config.json").exists():
            raise FileNotFoundError(
                f"DeBERTa-v3-large BERT model not found at {_BERT_DIR}. "
                "It may still be downloading."
            )

        logger.info("Loading Style-Bert-VITS2 GLaDOS model…")

        from style_bert_vits2.constants import Languages
        from style_bert_vits2.nlp import bert_models
        from style_bert_vits2.tts_model import TTSModel

        # Keep DeBERTa in fp16 to save ~400 MB RAM.
        # The VITS en_bert_proj layer expects float32 input, so we patch it
        # to cast the fp16 BERT output to float32 transparently.
        bert_models.load_model(Languages.EN, pretrained_model_name_or_path=str(_BERT_DIR))
        bert_models.load_tokenizer(Languages.EN, pretrained_model_name_or_path=str(_BERT_DIR))

        self._model = TTSModel(
            model_path=_MODEL_DIR / "model.safetensors",
            config_path=_MODEL_DIR / "config.json",
            style_vec_path=_MODEL_DIR / "style_vectors.npy",
            device="cpu",
        )
        # Force load now so net_g exists for patching (it's normally lazy)
        self._model.load()

        # Patch the BERT→VITS projection so fp16 BERT output auto-casts to float32.
        # net_g is name-mangled as _TTSModel__net_g in the private API.
        net_g = getattr(self._model, "_TTSModel__net_g", None)
        if net_g is not None and hasattr(net_g, "enc_p"):
            proj = net_g.enc_p.en_bert_proj
            _orig_fwd = proj.forward
            def _cast_fwd(x, _fwd=_orig_fwd):
                return _fwd(x.float())
            proj.forward = _cast_fwd
            logger.info("Patched en_bert_proj: fp16 BERT → float32 cast on-the-fly.")
        else:
            logger.warning("Could not find net_g.enc_p.en_bert_proj to patch.")

        self._loaded = True
        logger.info("Style-Bert-VITS2 GLaDOS ready (44100 Hz, 5 styles).")

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        style: str = "Neutral",
        style_weight: float = 1.0,
        noise: float = 0.6,
        noise_w: float = 0.8,
    ) -> dict:
        self._load()
        clean = _clean_tts_text(text)
        if not clean:
            return {"audio": b"", "visemes": [], "duration_ms": 0}

        # Clamp style to valid set
        if style not in STYLES:
            style = "Neutral"

        from style_bert_vits2.constants import Languages

        assert self._model is not None
        sr, audio_f32 = self._model.infer(
            text=clean,
            language=Languages.EN,
            speaker_id=0,
            style=style,
            style_weight=style_weight,
            noise=noise,
            noise_w=noise_w,
            length=1.0 / max(speed, 0.1),
        )
        self.sample_rate = sr

        # audio_f32 is float32; normalise and convert to int16
        peak = np.abs(audio_f32).max()
        if peak > 0:
            audio_f32 = audio_f32 / peak
        audio_i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_i16.tobytes())

        wav_bytes   = buf.getvalue()
        duration_ms = int(len(audio_i16) / sr * 1000)

        return {
            "audio":       wav_bytes,
            "visemes":     _letter_visemes(clean, duration_ms),
            "duration_ms": duration_ms,
        }
