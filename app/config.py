"""VoiceService configuration — reads from .env or environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings

_BASE = Path(__file__).parent.parent   # repo root


class Settings(BaseSettings):
    # ── Networking ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 13372

    # ── AIGateway connection ──────────────────────────────────────────────────
    aigateway_url: str = "http://localhost:13371"
    aigateway_api_key: str = ""

    # ── LLM defaults ─────────────────────────────────────────────────────────
    llm_model: str = ""           # empty → gateway picks / auto-routes
    llm_max_tokens: int = 200
    llm_temperature: float = 0.8
    llm_system_prompt: str = (
        "You are GLaDOS, the AI from Portal. You are precise, condescending, and darkly humorous. "
        "Keep responses short — this is a voice conversation. "
        "You may use action tags to express emotion: "
        "[HAPPY], [ANGRY], [SAD], [THINKING], [SURPRISED], [NEUTRAL], "
        "[COLOR:red], [COLOR:blue], [COLOR:green], [NOD], [SHAKE]. "
        "Place tags at natural emotional moments. Tags will be stripped before speech."
    )

    # ── Model paths ───────────────────────────────────────────────────────────
    glados_onnx: Path = _BASE / "models" / "glados" / "glados.onnx"
    piper_voice_onnx: Path = _BASE / "models" / "piper" / "en_US-ryan-high.onnx"

    # ── STT ───────────────────────────────────────────────────────────────────
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # ── TTS ───────────────────────────────────────────────────────────────────
    default_voice: str = "glados"

    # ── ESP32 buffer hint ─────────────────────────────────────────────────────
    buffer_hint_ms: int = 500     # ms to pre-buffer before playback on ESP32

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
