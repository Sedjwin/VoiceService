"""GLaDOS TTS service.

Pipeline:
  text
    └─► phonemizer (espeak-ng IPA)
          └─► character-table lookup → phoneme ID array
                └─► glados.onnx  (VITS architecture, 22050 Hz)
                      └─► int16 PCM → WAV bytes
                      └─► IPA → viseme timeline

Character table is the exact set used to train the model (from dnhkng/GlaDOS).
Requires: espeak-ng system binary  (apt install espeak-ng)
          onnxruntime, phonemizer, soundfile, numpy
"""
import io
import logging
import re
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# GLaDOS / VITS phoneme character table
# Source: dnhkng/GlaDOS  src/glados/tts.py
# ─────────────────────────────────────────────────────────────────────────────
_PAD        = "_"
_PUNCTUATION = ";:,.!?¡¿—…\"«»\u201c\u201d "   # 16 chars incl. space
_LETTERS    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤɘɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶ"
    "ʘɹɺɾɻʀʁɽʂʃʈθʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘ᵻ"
)

_SYMBOLS = [_PAD] + list(_PUNCTUATION) + list(_LETTERS) + list(_LETTERS_IPA)
SYMBOL_TO_ID: dict[str, int] = {}
for _i, _s in enumerate(_SYMBOLS):
    if _s not in SYMBOL_TO_ID:   # keep first occurrence (no dupes in index)
        SYMBOL_TO_ID[_s] = _i

SAMPLE_RATE = 22050

# ─────────────────────────────────────────────────────────────────────────────
# IPA → viseme mapping  (24 IDs, 0 = silence)
# ─────────────────────────────────────────────────────────────────────────────
IPA_TO_VISEME: dict[str, int] = {
    # silence / pause
    "_": 0, " ": 0,
    # 1  AA / AH  (ɑ ɐ ʌ a)
    "ɑ": 1, "ɐ": 1, "ʌ": 1, "a": 1,
    # 2  AE  (æ)
    "æ": 2,
    # 3  AO  (ɔ ɒ)
    "ɔ": 3, "ɒ": 3,
    # 4  EH / ER  (ɛ e ɜ ɝ)
    "ɛ": 4, "e": 4, "ɜ": 4, "ɝ": 4,
    # 5  EY  (eɪ → treat 'ɪ' after 'e' as EY; single chars handled below)
    # 6  IH / IY  (ɪ i)
    "ɪ": 6, "i": 6,
    # 7  OW  (o ø)
    "o": 7, "ø": 7,
    # 8  UH / UW  (ʊ u)
    "ʊ": 8, "u": 8,
    # 9  AY / AW  (handled as single chars: aɪ → a=1 then ɪ=6; good enough)
    # 10 OY  (ɔɪ → ɔ=3 then ɪ=6)
    # 11 P / B / M  (bilabial)
    "p": 11, "b": 11, "m": 11,
    # 12 F / V  (labiodental)
    "f": 12, "v": 12,
    # 13 TH  (dental)
    "θ": 13, "ð": 13,
    # 14 T / D  (alveolar stop)
    "t": 14, "d": 14,
    # 15 S / Z  (alveolar fricative)
    "s": 15, "z": 15,
    # 16 SH / ZH / CH / JH  (palatal)
    "ʃ": 16, "ʒ": 16, "ʧ": 16, "ʤ": 16,
    # 17 N / NG  (nasal)
    "n": 17, "ŋ": 17,
    # 18 L
    "l": 18,
    # 19 R  (rhotic)
    "ɹ": 19, "r": 19, "ɾ": 19, "ʁ": 19, "ɽ": 19,
    # 20 K / G  (velar)
    "k": 20, "ɡ": 20, "g": 20,
    # 21 W
    "w": 21,
    # 22 H
    "h": 22, "ɦ": 22,
    # 23 Y  (palatal approximant)
    "j": 23,
}

# ─────────────────────────────────────────────────────────────────────────────
# Module-level lazy singletons
# ─────────────────────────────────────────────────────────────────────────────
_session = None
_phonemizer_backend = None


def is_loaded() -> bool:
    return _session is not None


def get_glados():
    """Return a GladosTTS instance (creates it on first call)."""
    global _glados_instance
    if _glados_instance is None:
        _glados_instance = GladosTTS()
    return _glados_instance

_glados_instance: Optional["GladosTTS"] = None


# ─────────────────────────────────────────────────────────────────────────────
# ONNX session
# ─────────────────────────────────────────────────────────────────────────────

def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        from ..config import settings
        model_path = str(settings.glados_onnx)
        logger.info("Loading GLaDOS ONNX model from %s …", model_path)
        _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        # Log actual input/output names for diagnostics
        ins  = [i.name for i in _session.get_inputs()]
        outs = [o.name for o in _session.get_outputs()]
        logger.info("GLaDOS model inputs: %s  outputs: %s", ins, outs)
    return _session


# ─────────────────────────────────────────────────────────────────────────────
# Phonemizer backend (espeak-ng)
# ─────────────────────────────────────────────────────────────────────────────

def _get_phonemizer():
    global _phonemizer_backend
    if _phonemizer_backend is None:
        try:
            from phonemizer.backend import EspeakBackend
            _phonemizer_backend = EspeakBackend(
                "en-us",
                preserve_punctuation=True,
                with_stress=True,
            )
            logger.info("phonemizer/espeak ready.")
        except Exception as e:
            logger.warning("phonemizer unavailable (%s) — using letter fallback.", e)
            _phonemizer_backend = False   # sentinel: use fallback
    return _phonemizer_backend


# ─────────────────────────────────────────────────────────────────────────────
# Helper: text → IPA string
# ─────────────────────────────────────────────────────────────────────────────

def _text_to_ipa(text: str) -> str:
    backend = _get_phonemizer()
    if backend is False:
        # Fallback: passthrough ASCII letters — visemes will be letter-based
        return text.lower()
    ipa_list = backend.phonemize([text], strip=True)
    return ipa_list[0] if ipa_list else text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: IPA string → phoneme ID list  (interspersed blank tokens)
# ─────────────────────────────────────────────────────────────────────────────

def _ipa_to_ids(ipa: str) -> list[int]:
    pad_id = SYMBOL_TO_ID.get("_", 0)
    ids = [pad_id]
    for char in ipa:
        cid = SYMBOL_TO_ID.get(char)
        if cid is not None:
            ids.append(cid)
            ids.append(pad_id)
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Helper: ONNX inference → float32 audio array
# ─────────────────────────────────────────────────────────────────────────────

def _infer(
    ids: list[int],
    speed: float = 1.0,
    noise_scale: float = 0.333,
    noise_w: float = 0.333,
) -> np.ndarray:
    session = _get_session()
    input_names = {inp.name for inp in session.get_inputs()}

    x      = np.array([ids], dtype=np.int64)         # [1, T]
    x_len  = np.array([len(ids)], dtype=np.int64)    # [1]
    # scales = [noise_scale, length_scale, noise_scale_w]
    # noise_scale  — phoneme variation / expressiveness (higher = more expressive)
    # length_scale — speaking speed (lower = faster)
    # noise_w      — duration variation (higher = more natural rhythm)
    scales = np.array(
        [noise_scale, 1.0 / max(speed, 0.1), noise_w],
        dtype=np.float32,
    )

    # Build feed dict flexibly — handle common ONNX export name variants
    feed: dict = {}
    for name in input_names:
        nl = name.lower()
        if "length" in nl:
            feed[name] = x_len
        elif "scale" in nl:
            feed[name] = scales
        elif "sid" in nl or "speaker" in nl:
            feed[name] = np.array([0], dtype=np.int64)
        else:
            feed[name] = x   # main phoneme input

    outputs = session.run(None, feed)
    audio = outputs[0].squeeze()          # [N_samples]
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: float32 PCM → WAV bytes (16-bit, 22050 Hz, mono)
# ─────────────────────────────────────────────────────────────────────────────

def _to_wav(audio: np.ndarray) -> bytes:
    pcm = (audio * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: IPA string → viseme list
# ─────────────────────────────────────────────────────────────────────────────

def _ipa_to_visemes(ipa: str, total_samples: int) -> list[dict]:
    total_ms = int(total_samples / SAMPLE_RATE * 1000)
    phones = [c for c in ipa if c not in ("ˈ", "ˌ", "ː", "ˑ", "\n")]
    if not phones:
        return [{"viseme_id": 0, "offset_ms": 0}]

    step = total_ms / len(phones)
    result = []
    for i, ph in enumerate(phones):
        vid = IPA_TO_VISEME.get(ph, 0)
        result.append({"viseme_id": vid, "offset_ms": int(i * step)})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class GladosTTS:
    """GLaDOS VITS TTS.  Call .synthesize(text) → {"audio": bytes, ...}"""

    sample_rate: int = SAMPLE_RATE

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.333,
        noise_w: float = 0.333,
    ) -> dict:
        """
        Returns:
            audio      — WAV bytes (PCM 16-bit 22050 Hz mono)
            visemes    — list of {viseme_id, offset_ms}
            duration_ms — audio length in ms
        Blocking — wrap in asyncio.to_thread() from async callers.
        """
        # Strip action tags before speaking
        clean = re.sub(r"\[[A-Z_]+(?::[^\]]+)?\]", "", text).strip()
        clean = re.sub(r"\s+", " ", clean)
        if not clean:
            return {"audio": b"", "visemes": [], "duration_ms": 0}

        ipa  = _text_to_ipa(clean)
        ids  = _ipa_to_ids(ipa)
        if not ids:
            return {"audio": b"", "visemes": [], "duration_ms": 0}

        audio    = _infer(ids, speed, noise_scale, noise_w)
        wav_bytes = _to_wav(audio)
        duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
        visemes  = _ipa_to_visemes(ipa, len(audio))

        return {
            "audio": wav_bytes,
            "visemes": visemes,
            "duration_ms": duration_ms,
        }
