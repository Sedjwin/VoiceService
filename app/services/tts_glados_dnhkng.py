"""GLaDOS TTS — dnhkng neural ONNX pipeline.

Uses the models from https://github.com/dnhkng/GlaDOS (release 0.1):
  - phomenizer_en.onnx : neural G2P (grapheme-to-phoneme) model
  - glados.onnx        : VITS voice synthesis model
  - phoneme_to_id.pkl  : phoneme → VITS input ID mapping
  - lang_phoneme_dict.pkl / token_to_idx.pkl / idx_to_token.pkl

This is architecturally the same VITS synthesis as the Piper GLaDOS voice but
with a different training run and a neural phonemizer instead of espeak-ng.
Default inference parameters are more expressive (noise_scale 0.667 vs 0.333).

Requires: onnxruntime, numpy, models/glados_dnhkng/*
"""
from __future__ import annotations

import io
import logging
import re
import wave
from pathlib import Path
from pickle import load
from typing import Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)
ort.set_default_logger_severity(3)  # suppress verbose ORT startup noise

# ─────────────────────────────────────────────────────────────────────────────
# Model paths
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "glados_dnhkng"

_GLADOS_ONNX       = _MODEL_DIR / "glados.onnx"
_PHONEMIZER_ONNX   = _MODEL_DIR / "phomenizer_en.onnx"
_PHONEME_TO_ID     = _MODEL_DIR / "phoneme_to_id.pkl"
_LANG_PHONEME_DICT = _MODEL_DIR / "lang_phoneme_dict.pkl"
_TOKEN_TO_IDX      = _MODEL_DIR / "token_to_idx.pkl"
_IDX_TO_TOKEN      = _MODEL_DIR / "idx_to_token.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# IPA phoneme → 24-viseme ID  (same mapping as tts_glados.py)
# ─────────────────────────────────────────────────────────────────────────────
_IPA_VISEME: dict[str, int] = {
    "ɑ": 1, "ɐ": 1, "ɒ": 1,
    "æ": 2,
    "ʌ": 3, "ə": 3,
    "ɔ": 4,
    "aʊ": 5, "a": 5,
    "eɪ": 6, "e": 7,
    "ɛ": 7,
    "ɜ": 8, "ɝ": 8, "ɞ": 8,
    "ɪ": 9,
    "i": 10,
    "oʊ": 11, "o": 11,
    "ɔɪ": 12,
    "ʊ": 13,
    "u": 14,
    "b": 15, "p": 15, "m": 15,
    "f": 16, "v": 16,
    "θ": 17, "ð": 17,
    "d": 18, "t": 18, "n": 18,
    "k": 19, "g": 19, "ŋ": 19,
    "tʃ": 20, "dʒ": 20, "ʃ": 20, "ʒ": 20,
    "s": 21, "z": 21,
    "l": 22, "r": 22, "ɹ": 22,
    "w": 23, "j": 23,
}


def _ipa_visemes(phoneme_str: str, total_ms: int) -> list[dict]:
    """Map IPA phoneme string to timed viseme events (simple equal-time split)."""
    # Break into individual characters / digraphs
    chars = list(phoneme_str)
    if not chars:
        return [{"viseme_id": 0, "offset_ms": 0}]
    step = max(total_ms / len(chars), 1)
    result = []
    for i, ch in enumerate(chars):
        vid = _IPA_VISEME.get(ch, 0)
        result.append({"viseme_id": vid, "offset_ms": int(i * step)})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning  (same rules as the Piper GLaDOS service)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Neural phonemizer
# ─────────────────────────────────────────────────────────────────────────────

class _Phonemizer:
    """Neural G2P using phomenizer_en.onnx + lang_phoneme_dict.pkl."""

    _CHAR_REPEATS       = 3
    _MODEL_INPUT_LENGTH = 64

    _PUNC = set("().,:?!/– -")

    _PAD   = "_"
    _START = "<start>"
    _END   = "<end>"
    _SPECIAL = {"_", "<end>", "<en_us>"}

    def __init__(self) -> None:
        providers = [p for p in ort.get_available_providers()
                     if p not in ("TensorrtExecutionProvider", "CoreMLExecutionProvider")]
        self._session = ort.InferenceSession(str(_PHONEMIZER_ONNX), providers=providers)

        with _LANG_PHONEME_DICT.open("rb") as f:
            self._phoneme_dict: dict[str, str] = load(f)
        with _TOKEN_TO_IDX.open("rb") as f:
            self._token_to_idx: dict[str, int] = load(f)
        with _IDX_TO_TOKEN.open("rb") as f:
            self._idx_to_token: dict[int, str] = load(f)

        # Hard-code the canonical GLaDOS pronunciation
        self._phoneme_dict["glados"] = "ɡlˈɑːdɑːs"
        for p in self._PUNC:
            self._phoneme_dict[p] = p

    def _encode(self, word: str) -> list[int]:
        chars = [c for c in word.lower() for _ in range(self._CHAR_REPEATS)]
        seq = [self._token_to_idx[c] for c in chars if c in self._token_to_idx]
        return [self._token_to_idx[self._START], *seq, self._token_to_idx[self._END]]

    def _decode(self, ids: np.ndarray) -> str:
        tokens = [self._idx_to_token.get(int(i), "") for i in ids]
        return "".join(t for t in tokens if t not in self._SPECIAL)

    @staticmethod
    def _pad_batch(seqs: list[list[int]], length: int) -> np.ndarray:
        out = np.zeros((len(seqs), length), dtype=np.int64)
        for i, s in enumerate(seqs):
            n = min(len(s), length)
            out[i, :n] = s[:n]
        return out

    @staticmethod
    def _process_output(raw: list[np.ndarray]) -> list[np.ndarray]:
        ids = np.argmax(raw[0], axis=2)
        result = []
        for row in ids:
            # unique_consecutive
            mask = np.concatenate(([True], row[1:] != row[:-1]))
            row = row[mask]
            # remove padding (0)
            row = row[row != 0]
            # trim to end token (2)
            stops = np.where(row == 2)[0]
            if len(stops):
                row = row[: stops[0] + 1]
            result.append(row)
        return result

    def phonemize(self, text: str) -> str:
        """Convert a text string to an IPA phoneme string."""
        punc_re = re.compile(r"([().,:?!/\– -])")
        cleaned = "".join(c for c in text if c.isalnum() or c in self._PUNC)
        words = [w for w in re.split(punc_re, cleaned) if w]

        # Dictionary lookup first
        phonemes: dict[str, str | None] = {}
        for w in words:
            key = w.lower()
            if key in self._phoneme_dict:
                phonemes[w] = self._phoneme_dict[key]
            elif w.title() in self._phoneme_dict:
                phonemes[w] = self._phoneme_dict[w.title()]
            else:
                phonemes[w] = None

        # Neural model for unknown words
        unknowns = [w for w, p in phonemes.items() if p is None]
        if unknowns:
            batch = [self._encode(w) for w in unknowns]
            padded = self._pad_batch(batch, self._MODEL_INPUT_LENGTH)
            inp_name = self._session.get_inputs()[0].name
            outs = self._session.run(None, {inp_name: padded})
            ids_list = self._process_output(outs)
            for w, ids in zip(unknowns, ids_list):
                phonemes[w] = self._decode(ids)

        return "".join(phonemes.get(w) or w for w in words)


# ─────────────────────────────────────────────────────────────────────────────
# VITS synthesizer
# ─────────────────────────────────────────────────────────────────────────────

class _VITSSynthesizer:
    """Runs the dnhkng VITS glados.onnx model."""

    MAX_WAV = 32767.0

    def __init__(self) -> None:
        import json
        providers = [p for p in ort.get_available_providers()
                     if p not in ("TensorrtExecutionProvider", "CoreMLExecutionProvider")]
        self._session = ort.InferenceSession(str(_GLADOS_ONNX), providers=providers)

        with _PHONEME_TO_ID.open("rb") as f:
            self._phoneme_to_id: dict[str, list[int]] = load(f)

        with (_MODEL_DIR / "glados.json").open() as f:
            cfg = json.load(f)

        inf = cfg.get("inference", {})
        self.sample_rate  = cfg["audio"]["sample_rate"]
        self.noise_scale  = inf.get("noise_scale",  0.667)
        self.length_scale = inf.get("length_scale", 1.0)
        self.noise_w      = inf.get("noise_w",      0.8)

        self._PAD = "_"
        self._BOS = "^"
        self._EOS = "$"

    def _phonemes_to_ids(self, phoneme_str: str) -> list[int]:
        ids: list[int] = list(self._phoneme_to_id.get(self._BOS, [0]))
        for ph in phoneme_str:
            if ph in self._phoneme_to_id:
                ids.extend(self._phoneme_to_id[ph])
                ids.extend(self._phoneme_to_id.get(self._PAD, [0]))
        ids.extend(self._phoneme_to_id.get(self._EOS, [0]))
        return ids

    def synthesize_ids(
        self,
        phoneme_ids: list[int],
        length_scale: float,
        noise_scale: float,
        noise_w: float,
    ) -> np.ndarray:
        arr    = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        lengths = np.array([arr.shape[1]], dtype=np.int64)
        scales  = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)
        audio: np.ndarray = self._session.run(
            None,
            {"input": arr, "input_lengths": lengths, "scales": scales, "sid": None},
        )[0].squeeze((0, 1))
        return audio

    def synthesize(
        self,
        phoneme_str: str,
        length_scale: float,
        noise_scale: float,
        noise_w: float,
    ) -> np.ndarray:
        ids = self._phonemes_to_ids(phoneme_str)
        return self.synthesize_ids(ids, length_scale, noise_scale, noise_w)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface — mirrors GladosTTS in tts_glados.py
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional["GladosDnhkngTTS"] = None


def is_loaded() -> bool:
    return _instance is not None and _instance._loaded


def get_glados_dnhkng() -> "GladosDnhkngTTS":
    global _instance
    if _instance is None:
        _instance = GladosDnhkngTTS()
    return _instance


def _wav_duration_ms(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return int(w.getnframes() / w.getframerate() * 1000)


class GladosDnhkngTTS:
    """GLaDOS TTS via dnhkng neural ONNX pipeline.

    Call .synthesize(text) → {"audio": bytes, "visemes": [...], "duration_ms": int}
    """

    sample_rate: int = 22050

    def __init__(self) -> None:
        self._phonemizer: Optional[_Phonemizer] = None
        self._synth: Optional[_VITSSynthesizer] = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not _GLADOS_ONNX.exists():
            raise FileNotFoundError(f"dnhkng GLaDOS model not found: {_GLADOS_ONNX}")
        if not _PHONEMIZER_ONNX.exists():
            raise FileNotFoundError(f"dnhkng phonemizer not found: {_PHONEMIZER_ONNX}")
        logger.info("Loading dnhkng GLaDOS models from %s ...", _MODEL_DIR)
        self._phonemizer = _Phonemizer()
        self._synth      = _VITSSynthesizer()
        self.sample_rate = self._synth.sample_rate
        self._loaded     = True
        logger.info("dnhkng GLaDOS TTS ready  (sample_rate=%d).", self.sample_rate)

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> dict:
        self._load()
        clean = _clean_tts_text(text)
        if not clean:
            return {"audio": b"", "visemes": [], "duration_ms": 0}

        length_scale = 1.0 / max(speed, 0.1)

        assert self._phonemizer is not None
        assert self._synth is not None

        phoneme_str = self._phonemizer.phonemize(clean)
        audio_f32   = self._synth.synthesize(phoneme_str, length_scale, noise_scale, noise_w)

        # Convert float32 → int16 PCM
        audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * _VITSSynthesizer.MAX_WAV).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(self.sample_rate)
            wav_out.writeframes(audio_i16.tobytes())

        wav_bytes   = buf.getvalue()
        duration_ms = _wav_duration_ms(wav_bytes)

        return {
            "audio":       wav_bytes,
            "visemes":     _ipa_visemes(phoneme_str, duration_ms),
            "duration_ms": duration_ms,
        }
