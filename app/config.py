"""VoiceService configuration — pure audio service, no LLM knowledge."""
from pathlib import Path
from pydantic_settings import BaseSettings

_BASE = Path(__file__).parent.parent   # repo root


class Settings(BaseSettings):
    # ── Networking ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 13372

    # ── Model paths ───────────────────────────────────────────────────────────
    glados_onnx: Path = _BASE / "models" / "glados" / "glados.onnx"
    piper_voice_onnx: Path = _BASE / "models" / "piper" / "en_US-ryan-high.onnx"
    jarvis_voice_onnx: Path = _BASE / "models" / "piper" / "en_GB-alan-medium.onnx"
    tars_voice_onnx: Path = _BASE / "models" / "piper" / "en_US-hfc_male-medium.onnx"

    # ── STT ───────────────────────────────────────────────────────────────────
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # ── TTS ───────────────────────────────────────────────────────────────────
    default_voice: str = "glados"

    # ── ESP32 buffer hint ─────────────────────────────────────────────────────
    # Bytes to pre-buffer before I2S playback to absorb WiFi jitter (0.5 s @ 22050 Hz 16-bit)
    buffer_hint_ms: int = 500

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
