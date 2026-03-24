# VoiceService

Standalone STT/TTS microservice. Runs on Raspberry Pi 5. No auth, no LLM calls — pure speech I/O. All orchestration is handled by AIGateway or AgentManager.

## Voices

| ID | Name | Character |
|----|------|-----------|
| `glados`   | **GLaDOS**          | Aperture Science AI (Portal) — precise, condescending, darkly humorous |
| `hal`      | **HAL 9000**        | Sentient computer (2001: A Space Odyssey) — calm, eerily polite |
| `k9`       | **K-9**             | Robot dog (Doctor Who) — clipped, mechanical, loyal |
| `k9v2`     | **K-9 v2**          | K-9 alternate training — slightly different cadence |
| `jarvis`   | **JARVIS**          | Tony Stark's AI (Marvel MCU) — warm, articulate British male |
| `wheatley` | **Wheatley**        | Personality core (Portal 2) — bumbling, excitable, well-meaning |
| `data`     | **Commander Data**  | Android officer (Star Trek: TNG) — measured, precise, literal |

Per-voice speed and pitch are tunable via the admin UI at `/admin.html` and persisted to `data/voice_settings.json`. Models can be individually loaded/unloaded to manage RAM.

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

Runs on port **8002** (internal) / **13372** (external, HTTPS via Caddy).

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

### `GET /voices` — List voices
Returns available voices with their loaded status, character info, and supported parameters.

---

### `GET /voices/{voice_id}/settings` — Get per-voice tuning
Returns current speed, pitch, and other tuning parameters for a voice.

### `PUT /voices/{voice_id}/settings` — Update per-voice tuning
Persists tuning changes to `data/voice_settings.json`.

```json
{ "speed": 1.1, "pitch_shift": 0.0 }
```

---

### `GET /models` — List models and load status
Shows which models are currently loaded in memory.

### `POST /models/{voice_id}/load` — Force-load a model
Loads the model into memory immediately.

### `POST /models/{voice_id}/unload` — Unload a model
Frees the model from RAM (will reload on next synthesis request).

### `POST /models/{voice_id}/interrupt` — Block next synthesis
Causes the next TTS request for this voice to return `503`. Used for testing or graceful interruption.

---

### `GET /stats` — System stats
Returns CPU usage, RAM usage, and CPU temperature.

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

## Text Preprocessing (`_clean_tts_text`)

All text passes through a cleaning stage before being sent to espeak-ng. Applied in order:

### Prosody normalisation
Converts punctuation that espeak-ng ignores into forms it does understand:

| Input | Becomes | Effect |
|-------|---------|--------|
| `…` (U+2026) | `...` | Normalise Unicode ellipsis to ASCII so espeak handles it |
| `...` / `....` | `. ` | Single clean sentence-boundary pause (espeak's behaviour for raw `...` is version-dependent) |
| `——` / `––` (2+ dashes) | `. ` | Sentence-boundary pause — dramatic beat |
| `—` / `–` (single dash) | `, ` | Phrase-level pause — mid-sentence beat |

### Markdown stripping
Removes markup that espeak would read aloud literally:

- Bold/italic (`*text*`, `_text_`) → inner text only
- Strikethrough (`~~text~~`) → inner text only
- Headers (`# Title`) → text only
- Fenced code blocks (` ``` `) → removed entirely
- Inline code (`` `code` ``) → inner text only
- Block quotes (`> text`) → text only
- Markdown links/images → link text only
- Table pipes `|` → spaces
- Horizontal rules (`---`) → removed
- Remaining symbols espeak reads literally: `* _ \ ^ ~ \` < > { #` → removed
- Multiple newlines → `. ` (paragraph pause)
- Remaining newlines → space
- Runs of whitespace → single space

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
