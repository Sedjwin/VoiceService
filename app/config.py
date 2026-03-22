"""VoiceService configuration — pure audio service, no LLM knowledge."""
from pathlib import Path
from pydantic_settings import BaseSettings

_BASE = Path(__file__).parent.parent   # repo root
_PIPER = _BASE / "models" / "piper"


class Settings(BaseSettings):
    # ── Networking ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 13372

    # ── Model paths ───────────────────────────────────────────────────────────
    glados_onnx:    Path = _BASE / "models" / "glados" / "glados.onnx"
    hal_onnx:       Path = _PIPER / "hal.onnx"
    k9_onnx:        Path = _PIPER / "k9_model.onnx"
    k9v2_onnx:      Path = _PIPER / "k9_2449_model.onnx"
    jarvis_onnx:    Path = _PIPER / "jarvis-high.onnx"
    wheatley_onnx:  Path = _PIPER / "wheatley1.onnx"
    data_onnx:      Path = _PIPER / "data.onnx"

    # ── STT ───────────────────────────────────────────────────────────────────
    whisper_model:        str = "base.en"
    whisper_device:       str = "cpu"
    whisper_compute_type: str = "int8"

    # ── TTS ───────────────────────────────────────────────────────────────────
    default_voice: str = "glados"

    # ── ESP32 buffer hint ─────────────────────────────────────────────────────
    buffer_hint_ms: int = 500

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
