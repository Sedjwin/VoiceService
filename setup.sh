#!/usr/bin/env bash
# VoiceService setup — run once on a fresh system
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║        VoiceService Setup            ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ───────────────────────────────────────────────────────────────────────────
# 1. System dependencies
# ───────────────────────────────────────────────────────────────────────────
echo "→ Installing system dependencies (espeak-ng, libsndfile)…"
sudo apt-get update -qq
sudo apt-get install -y espeak-ng libsndfile1
echo "  ✓ System deps installed"

# ───────────────────────────────────────────────────────────────────────────
# 2. Python virtualenv
# ───────────────────────────────────────────────────────────────────────────
echo "→ Setting up Python virtual environment…"
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install python3.11+ first."
    exit 1
fi

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created .venv"
fi

source .venv/bin/activate
echo "→ Installing Python dependencies…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  ✓ Python deps installed"

# ───────────────────────────────────────────────────────────────────────────
# 3. Download model files
# ───────────────────────────────────────────────────────────────────────────
echo "→ Downloading TTS/STT models…"
echo "  (GLaDOS ~32 MB, Piper ATLAS ~64 MB — may take a while)"
python3 download_models.py
echo "  ✓ Models ready"

# ───────────────────────────────────────────────────────────────────────────
# 4. Copy .env if not present
# ───────────────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ✓ Created .env from .env.example — edit it to set AIGATEWAY_API_KEY"
fi

# ───────────────────────────────────────────────────────────────────────────
# 5. Systemd service
# ───────────────────────────────────────────────────────────────────────────
SERVICE_FILE="$SCRIPT_DIR/voiceservice.service"
if [ -f "$SERVICE_FILE" ] && command -v systemctl &>/dev/null; then
    echo "→ Installing systemd service…"
    sudo cp "$SERVICE_FILE" /etc/systemd/system/voiceservice.service
    sudo systemctl daemon-reload
    sudo systemctl enable voiceservice.service
    echo "  ✓ voiceservice.service installed and enabled"
    echo "  Start now:  sudo systemctl start voiceservice"
    echo "  Logs:       sudo journalctl -u voiceservice -f"
fi

deactivate

# ───────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║        Setup Complete!               ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "  To start:      ./start.sh"
echo "  API docs:      http://localhost:13372/docs"
echo "  Health check:  http://localhost:13372/health"
echo ""
echo "  IMPORTANT: Edit .env and set AIGATEWAY_API_KEY to an"
echo "  agent Bearer token from your AIGateway panel."
echo ""
