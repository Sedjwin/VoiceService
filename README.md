# VoiceService

Standalone STT/TTS microservice for [AIGateway](https://github.com/sedjwin/AIGateway).
Runs on Raspberry Pi 5, optimised for ESP32-S3 clients with I2S microphones and speakers.

## Voices

| ID | Name | Source | Character |
|----|------|--------|-----------|
| `glados` | **GLaDOS** | dnhkng/GlaDOS ONNX VITS | Aperture Science AI (Portal) — precise, condescending, darkly humorous |
| `atlas`  | **ATLAS**  | Piper en_US-ryan-high    | Cooperative android (Portal 2) — clear, professional AI-assistant tone |

## Architecture

```
ESP32-S3 mic (WAV 16 kHz)
        │
        ▼
POST /voice/chat
        │
        ├─► Whisper STT  →  transcript
        │
        ├─► AIGateway /v1/chat/completions  →  LLM response text
        │         (brings agent permissions, smart routing, logging)
        │
        ├─► Action tag parser  →  [HAPPY], [ANGRY], [COLOR:red] …
        │
        └─► GLaDOS / ATLAS TTS  →  WAV audio + viseme timeline
                │
                ▼
        JSON response to ESP32
```

## Installation

```bash
./setup.sh        # installs deps, downloads models, sets up systemd
```

Requires Python 3.11+. `espeak-ng` is installed automatically.

## Running

```bash
./start.sh                          # development (--reload)
sudo systemctl start voiceservice   # production (systemd)
sudo journalctl -u voiceservice -f  # logs
```

Runs on port **13372**.

## Configuration

Copy `.env.example` → `.env` and set:

```bash
AIGATEWAY_API_KEY=<your-agent-bearer-token>  # from AIGateway Agents panel
AIGATEWAY_URL=http://localhost:13371          # default
```

## API Reference

### `POST /stt` — Speech to Text
Send raw WAV bytes in the request body.

```bash
curl -X POST http://localhost:13372/stt \
     -H "Content-Type: audio/wav" \
     --data-binary @recording.wav
```

Response:
```json
{"text": "Hello there", "language": "en", "duration_ms": 320}
```

---

### `POST /tts` — Text to Speech (base64 response)
```bash
curl -X POST http://localhost:13372/tts \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello test subject", "voice": "glados"}'
```

Response:
```json
{
  "voice": "glados",
  "audio": "<base64 WAV>",
  "audio_format": "wav",
  "sample_rate": 22050,
  "duration_ms": 1200,
  "buffer_bytes": 22050,
  "visemes": [
    {"viseme_id": 0, "offset_ms": 0},
    {"viseme_id": 22, "offset_ms": 45},
    ...
  ]
}
```

`buffer_bytes` = bytes to pre-buffer before starting I2S playback (0.5 s).

---

### `POST /tts/raw` — Text to Speech (raw WAV)
Same request body as `/tts`, returns `audio/wav` bytes directly.

---

### `POST /voice/chat` — Full Pipeline
Multipart form upload:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `audio` | file | — | WAV from ESP32 mic |
| `voice` | string | `glados` | `glados` or `atlas` |
| `speed` | float | `1.0` | TTS speed (0.25–4.0) |
| `model` | string | `` | AIGateway model ID (empty = auto) |
| `api_key` | string | env | Agent Bearer token |
| `system_prompt` | string | `` | Override LLM system prompt |

Response:
```json
{
  "input_text":    "user's transcribed speech",
  "response_text": "full LLM response (may have action tags)",
  "clean_text":    "spoken text (tags stripped)",
  "actions":       [{"type": "expression", "value": "happy"}],
  "voice":         "glados",
  "model_used":    "qwen2.5:1.5b",
  "audio":         "<base64 WAV>",
  "audio_format":  "wav",
  "sample_rate":   22050,
  "duration_ms":   1800,
  "buffer_bytes":  22050,
  "visemes":       [...],
  "pipeline_ms":   940
}
```

---

### `POST /voice/tts-chat` — Text Chat (no STT)
Like `/voice/chat` but accepts a text `text` field instead of audio.
Useful for testing LLM + TTS without a microphone.

---

### `GET /voices` — List voices
Returns available voices with their loaded status, character info, and supported parameters.

---

### `GET /activity` — Current state and operation log
```bash
curl http://localhost:13372/activity
```

Response:
```json
{
  "state": "idle",
  "since": "2025-03-17T12:00:00",
  "preview": null,
  "voice": "glados",
  "log": [
    {
      "id": 1,
      "type": "tts",
      "started_at": "2025-03-17T11:59:30",
      "finished_at": "2025-03-17T11:59:31",
      "duration_ms": 340
    }
  ]
}
```

States: `idle` · `stt` · `tts`

---

### `GET /health` — Service health

---

## Action Tags

The LLM system prompt instructs the model to include action tags.
These are stripped before speech synthesis and returned in `actions[]`.

| Tag | Type | Value |
|-----|------|-------|
| `[HAPPY]` | expression | happy |
| `[ANGRY]` | expression | angry |
| `[SAD]` | expression | sad |
| `[THINKING]` | expression | thinking |
| `[SURPRISED]` | expression | surprised |
| `[NEUTRAL]` | expression | neutral |
| `[NOD]` | gesture | nod |
| `[SHAKE]` | gesture | shake |
| `[BLINK]` | gesture | blink |
| `[COLOR:red]` | color | red |
| `[COLOR:blue]` | color | blue |
| `[COLOR:green]` | color | green |

ESP32 interprets these to control LEDs, servos, or display expressions.

---

## Viseme IDs (24-set)

| ID | Name | Phonemes |
|----|------|---------|
| 0 | silence | (pause) |
| 1 | AA/AH | ɑ ɐ ʌ a |
| 2 | AE | æ |
| 3 | AO | ɔ ɒ |
| 4 | EH/ER | ɛ e ɜ ɝ |
| 5 | EY | eɪ |
| 6 | IH/IY | ɪ i |
| 7 | OW | o ø |
| 8 | UH/UW | ʊ u |
| 9 | AY/AW | aɪ aʊ |
| 10 | OY | ɔɪ |
| 11 | P/B/M | p b m |
| 12 | F/V | f v |
| 13 | TH | θ ð |
| 14 | T/D | t d |
| 15 | S/Z | s z |
| 16 | SH/ZH/CH | ʃ ʒ |
| 17 | N/NG | n ŋ |
| 18 | L | l |
| 19 | R | ɹ r ɾ |
| 20 | K/G | k ɡ |
| 21 | W | w |
| 22 | H | h |
| 23 | Y | j |

---

## ESP32 Integration Notes

- Send audio as **WAV PCM 16-bit 16 kHz mono** (standard I2S capture format)
- Receive JSON response; decode base64 `audio` field to WAV bytes
- Buffer `buffer_bytes` worth of audio before starting I2S output to absorb WiFi jitter
- Max recommended clip: ~10 seconds of speech input

```cpp
// Minimal ESP32 Arduino sketch outline
WiFiClient client;
client.connect("192.168.1.x", 13372);
client.println("POST /stt HTTP/1.1");
client.println("Content-Type: audio/wav");
client.print("Content-Length: "); client.println(audioLen);
client.println("Connection: close");
client.println();
client.write(audioBuffer, audioLen);
// Read JSON response...
```

## Ports

| Service | Port |
|---------|------|
| AIGateway | 13371 |
| VoiceService | 13372 |
