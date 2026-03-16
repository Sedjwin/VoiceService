#!/usr/bin/env bash
# Start VoiceService in development mode (--reload enabled)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run ./setup.sh first."
    exit 1
fi

source .venv/bin/activate

HOST="${VOICESERVICE_HOST:-0.0.0.0}"
PORT="${VOICESERVICE_PORT:-13372}"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  VoiceService starting on http://${HOST}:${PORT}"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  STT:       POST /stt"
echo "║  TTS:       POST /tts         (JSON → base64 WAV)"
echo "║  TTS raw:   POST /tts/raw     (JSON → WAV bytes)"
echo "║  Pipeline:  POST /voice/chat  (multipart → full response)"
echo "║  Text chat: POST /voice/tts-chat"
echo "║  Voices:    GET  /voices"
echo "║  Docs:      http://localhost:${PORT}/docs"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

exec uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload \
    --log-level info
