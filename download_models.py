#!/usr/bin/env python3
"""Download VoiceService model files.

Downloads:
  1. GLaDOS ONNX TTS model  — from dnhkng/GlaDOS on HuggingFace (~32 MB)
  2. ATLAS Piper voice       — en_US-ryan-high.onnx from rhasspy/piper-voices (~64 MB)

Whisper STT model (base.en, ~142 MB) is auto-downloaded by faster-whisper
on first use — no action needed here.

Usage:
    python3 download_models.py
    python3 download_models.py --skip-glados   # only download Piper
    python3 download_models.py --skip-piper    # only download GLaDOS
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR  = Path(__file__).parent / "models"
GLADOS_DIR  = MODELS_DIR / "glados"
PIPER_DIR   = MODELS_DIR / "piper"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _progress(downloaded: int, chunk: int, total: int):
    if total > 0:
        pct = downloaded * 100 // total
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {downloaded // 1024 // 1024} MB / {total // 1024 // 1024} MB", end="", flush=True)


def _dl(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {url.split('/')[-1]}")
    urllib.request.urlretrieve(url, dest, _progress)
    print()   # newline after progress bar


# ─────────────────────────────────────────────────────────────────────────────
# GLaDOS
# ─────────────────────────────────────────────────────────────────────────────

def download_glados():
    """Download glados.onnx from dnhkng/GlaDOS on HuggingFace."""
    onnx_path = GLADOS_DIR / "glados.onnx"

    if onnx_path.exists():
        print(f"GLaDOS model already present at {onnx_path}  (skipping)")
        return

    print("\n── GLaDOS ONNX model ─────────────────────────────────────────────────")
    GLADOS_DIR.mkdir(parents=True, exist_ok=True)

    # Public mirror (rokeya71/VITS-Piper-GlaDOS-en-onnx) — original dnhkng/GlaDOS is gated
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        print("  Fetching from HuggingFace (rokeya71/VITS-Piper-GlaDOS-en-onnx) …")
        cached = hf_hub_download(
            repo_id="rokeya71/VITS-Piper-GlaDOS-en-onnx",
            filename="glados.onnx",
        )
        shutil.copy(cached, onnx_path)
        print(f"  ✓ glados.onnx  →  {onnx_path}")
        return
    except Exception as e:
        print(f"  huggingface_hub failed ({e}), trying direct URL …")

    # Fallback: direct HTTPS
    url = "https://huggingface.co/rokeya71/VITS-Piper-GlaDOS-en-onnx/resolve/main/glados.onnx"
    _dl(url, onnx_path)
    print(f"  ✓ glados.onnx  →  {onnx_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Piper ATLAS voice  (en_US-ryan-high)
# ─────────────────────────────────────────────────────────────────────────────

_PIPER_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    "/en/en_US/ryan/high"
)

def download_piper():
    """Download en_US-ryan-high.onnx + .json from rhasspy/piper-voices."""
    onnx_path = PIPER_DIR / "en_US-ryan-high.onnx"
    json_path = PIPER_DIR / "en_US-ryan-high.onnx.json"

    if onnx_path.exists() and json_path.exists():
        print(f"Piper ATLAS voice already present at {PIPER_DIR}  (skipping)")
        return

    print("\n── Piper ATLAS voice (en_US-ryan-high) ───────────────────────────────")
    PIPER_DIR.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        _dl(f"{_PIPER_BASE}/en_US-ryan-high.onnx", onnx_path)
    if not json_path.exists():
        _dl(f"{_PIPER_BASE}/en_US-ryan-high.onnx.json", json_path)

    print(f"  ✓ ATLAS voice  →  {PIPER_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Piper JARVIS voice  (en_GB-alan-medium)
# ─────────────────────────────────────────────────────────────────────────────

_JARVIS_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    "/en/en_GB/alan/medium"
)

def download_jarvis():
    """Download en_GB-alan-medium.onnx + .json from rhasspy/piper-voices."""
    onnx_path = PIPER_DIR / "en_GB-alan-medium.onnx"
    json_path = PIPER_DIR / "en_GB-alan-medium.onnx.json"

    if onnx_path.exists() and json_path.exists():
        print(f"Piper JARVIS voice already present at {PIPER_DIR}  (skipping)")
        return

    print("\n── Piper JARVIS voice (en_GB-alan-medium) ────────────────────────────")
    PIPER_DIR.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        _dl(f"{_JARVIS_BASE}/en_GB-alan-medium.onnx", onnx_path)
    if not json_path.exists():
        _dl(f"{_JARVIS_BASE}/en_GB-alan-medium.onnx.json", json_path)

    print(f"  ✓ JARVIS voice  →  {PIPER_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Piper TARS voice  (TARS-AI dedicated model)
# ─────────────────────────────────────────────────────────────────────────────

_TARS_BASE = (
    "https://raw.githubusercontent.com/TARS-AI-Community/TARS-AI/V2"
    "/src/character/TARS/voice"
)

def download_tars():
    """Download dedicated TARS ONNX + .json from TARS-AI community repo."""
    onnx_path = PIPER_DIR / "en_US-tars-ai-medium.onnx"
    json_path = PIPER_DIR / "en_US-tars-ai-medium.onnx.json"

    if onnx_path.exists() and json_path.exists():
        print(f"Piper TARS voice already present at {PIPER_DIR}  (skipping)")
        return

    print("\n── Piper TARS voice (TARS-AI dedicated model) ───────────────────────")
    PIPER_DIR.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        _dl(f"{_TARS_BASE}/TARS.onnx", onnx_path)
    if not json_path.exists():
        _dl(f"{_TARS_BASE}/TARS.onnx.json", json_path)

    print(f"  ✓ TARS voice  →  {PIPER_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Piper TERMINATOR voice (robotic fallback profile)
# ─────────────────────────────────────────────────────────────────────────────

_TERMINATOR_BASE = (
    "https://huggingface.co/campwill/HAL-9000-Piper-TTS/resolve/main"
)

def download_terminator():
    """Download a robot-style ONNX profile for Terminator voice slot."""
    onnx_path = PIPER_DIR / "en_US-terminator-hal-medium.onnx"
    json_path = PIPER_DIR / "en_US-terminator-hal-medium.onnx.json"

    if onnx_path.exists() and json_path.exists():
        print(f"Piper TERMINATOR voice already present at {PIPER_DIR}  (skipping)")
        return

    print("\n── Piper TERMINATOR voice (robotic HAL profile) ─────────────────────")
    PIPER_DIR.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        _dl(f"{_TERMINATOR_BASE}/hal.onnx", onnx_path)
    if not json_path.exists():
        _dl(f"{_TERMINATOR_BASE}/hal.onnx.json", json_path)

    print(f"  ✓ TERMINATOR voice  →  {PIPER_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download VoiceService models")
    parser.add_argument("--skip-glados",  action="store_true")
    parser.add_argument("--skip-piper",   action="store_true")
    parser.add_argument("--skip-jarvis",  action="store_true")
    parser.add_argument("--skip-tars",    action="store_true")
    parser.add_argument("--skip-terminator", action="store_true")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════╗")
    print("║   VoiceService — Model Download  ║")
    print("╚══════════════════════════════════╝")

    if not args.skip_glados:
        download_glados()
    if not args.skip_piper:
        download_piper()
    if not args.skip_jarvis:
        download_jarvis()
    if not args.skip_tars:
        download_tars()
    if not args.skip_terminator:
        download_terminator()

    print("\n── Whisper STT ───────────────────────────────────────────────────────")
    print("  Whisper base.en (~142 MB) downloads automatically on first STT call.")
    print("  Pre-download:  python3 -c \"from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')\"")

    print("\n╔══════════════════════════════════╗")
    print("║       All models ready!          ║")
    print("╚══════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
