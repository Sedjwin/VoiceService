"""Shared utilities for VoiceService.

VoiceService is a pure audio-conversion service — it does NOT call AIGateway
or any LLM. All orchestration (STT → LLM → TTS) is done by AIGateway, which
calls this service's /stt and /tts endpoints as internal utilities.

This module provides helpers used by the routers.
"""
import re

_ACTION_RE = re.compile(r"\[([A-Z_]+(?::[a-z0-9]+)?)\]")

BUFFER_SAMPLE_RATE = 22050  # Hz


def buffer_bytes(sample_rate: int = BUFFER_SAMPLE_RATE, hint_ms: int = 500) -> int:
    """Bytes to pre-buffer before ESP32 I2S playback (16-bit mono)."""
    return int(sample_rate * hint_ms / 1000 * 2)
