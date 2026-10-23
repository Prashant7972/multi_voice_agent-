# Apex Voice AI: Multilingual Local Voice Agent 🎙️

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/LLM-Gemma_3_4B-purple.svg)](https://ollama.ai)
[![Whisper](https://img.shields.io/badge/STT-faster--whisper-orange.svg)](https://github.com/SYSTRAN/faster-whisper)
[![Edge-TTS](https://img.shields.io/badge/TTS-Edge--TTS-green.svg)](https://github.com/rany2/edge-tts)
[![Tests Passing](https://img.shields.io/badge/tests-136%20passed-brightgreen.svg)]()

An enterprise-grade, low-latency, fully private local voice agent running on your own hardware with **dynamic turn-by-turn multilingual mirroring**, **Twilio telephony integration**, **RAG knowledge retrieval**, **multi-agent persona switching**, and an **interactive glassmorphic Web UI**.

```text
 ┌─────────────────┐       ┌─────────────────┐       ┌───────────────────┐       ┌─────────────────┐
 │   Microphone /  │ ────▶ │  Silero VAD v5  │ ────▶ │  faster-whisper   │ ────▶ │  Ollama Gemma 3 │
 │   Twilio Phone  │       │  Adaptive Floor │       │  Indic Morphemes  │       │  Spoken Brevity │
 └─────────────────┘       └─────────────────┘       └───────────────────┘       └─────────────────┘
                                                                                          │
 ┌─────────────────┐       ┌─────────────────┐       ┌───────────────────┐                │
 │ Speaker / Phone │ ◀──── │  Edge-TTS Voice │ ◀──── │ Dynamic Language  │ ◀──────────────┘
 │ Earpiece / Web  │       │  Piper Fallback │       │ Script Alignment  │
 └─────────────────┘       └─────────────────┘       └───────────────────┘
```

---

## 🌟 Key Highlights

### 1. 🌐 Dynamic Turn-by-Turn Multilingual Mirroring
* **No Rigid Language Lock**: The assistant adapts fluidly turn-by-turn to whatever language you speak:
  - Speak in **Marathi (मराठी)** ➔ Transcribed in authentic Devanagari script, Gemma 3 replies in Marathi, and Edge-TTS speaks in Marathi (`mr-IN-AarohiNeural`).
  - Switch to **English** ➔ Instantly responds in English audio (`en-US-AvaNeural`).
  - Speak in **Hindi (हिन्दी)** ➔ Responds in fluent Hindi (`hi-IN-MadhurNeural`).
  - Seamless support across **28+ languages** (Spanish, French, German, Japanese, Chinese, Arabic, Portuguese, Tamil, Telugu, and more).
* **Indic Script & Morpheme Disambiguation**: Solves Whisper's acoustic bias where Marathi speech was misclassified as Hindi or English. Automatically inspects Devanagari morphemes (`आहे`, `नाही`, `माझं`, `तुझ्या`, `मला`, `हवी`, `विषयी`, `बोलायचे`) and re-decodes with native acoustic prompts to prevent grammatical distortion.
* **Transliteration Guard**: Detects Latin-transliterated Indic phrases (e.g. *"tumhala mazha vishayi..."*) and restores native Devanagari script.

### 2. ⚡ Lightning-Fast Turnaround (~1.8s End-to-End)
* **Sub-Second STT**: CTranslate2 `int8` with `temperature=0.0` (eliminates 5x retry fallback delay) and `cpu_threads=4`.
* **Crisp LLM Generation**: Constrained to a 42-token voice budget (~25 words) for natural, punchy 1–2 sentence phone-like conversational turns without CPU lag.
* **Adaptive VAD**: Rolling noise-floor baseline with dynamic turn-taking thresholds (**750ms** for full sentences, **950ms** for short utterances).

### 3. 🧠 Gemma 3 4B Intelligence & Cloud Fallback
* **Primary Engine**: Runs local **Gemma 3 4B** (`gemma3:4b`) or **Qwen 2.5 Coder 3B** via Ollama on port `11434` / `11435`.
* **Script-Locked Directives**: Context-anchored prompt headers prevent language bleed or phonetic hallucinations (e.g. preventing *"माझं"* from becoming *"mazes"*).
* **Multi-Key Gemini Fallback**: Persistent HTTP keep-alive connection pooling with automatic multi-key round-robin rotation for zero-budget thinking mode.

### 4. 📞 Twilio Voice Calling (Inbound & Outbound)
* **Inbound Calling**: Dial your Twilio phone number from any mobile or landline to speak directly with the local agent.
* **Outbound Dialing**: Initiate automated calls straight from the Web UI dialer.
* **Bidirectional WebSocket Streaming**: Real-time on-the-fly transcoding between 8kHz mu-law and 16kHz linear PCM.

### 5. 🎙️ Modern Glassmorphic Web UI
* **Interactive Audio Replay**:
  - **`▶ Replay Your Voice`**: Listen back to your recorded microphone captures to verify clarity and background noise.
  - **`▶ Replay Agent Audio`**: Re-listen to the synthesized AI voice reply at any time.
* **Telemetry Badges**: Real-time latency breakdown chips (`⚡ Total` • `🎙️ STT` • `🧠 LLM` • `🔊 TTS`).
* **Visualizer Orb**: Glowing, pulsing 3D CSS orb with real-time dynamic states (*Listening*, *Transcribing*, *Thinking*, *Speaking*).
* **Multi-Agent & RAG Management**: Create custom agent personas, ingest knowledge base documents, and schedule outbound campaigns.

---

## 📂 Architecture & Directory Structure

```text
local-voice-agent/
├── app/
│   ├── main.py                  # CLI entrypoint, argument parser, and runtime launcher
│   ├── server.py                # FastAPI server (Web UI, /api/chat, /twiml/stream, JWT Auth, RAG)
│   ├── stt.py                   # faster-whisper STT with dynamic Indic morpheme disambiguation
│   ├── llm.py                   # Ollama (Gemma 3 4B) & Gemini API client with keep-alive pooling
│   ├── tts.py                   # Multilingual Edge-TTS (28+ neural voices) & Piper ONNX fallback
│   ├── audio.py                 # ALSA audio capture, Silero VAD v5, dynamic ambient noise floor
│   ├── agents.py                # Multi-agent persona & lifecycle manager
│   ├── knowledge_base.py        # RAG vector storage, document chunking & similarity search
│   ├── twilio_handler.py        # Twilio WebSocket streaming, 8kHz mu-law audio transcoding
│   ├── analytics.py             # Telemetry tracking, interaction metrics, and call logs
│   ├── auth_manager.py          # HMAC-SHA256 JWT tokens & 6-digit email OTP verification
│   ├── config.py                # Hyperparameters, Ollama auto-detection, port mappings
│   └── templates/
│       └── index.html           # Single-page glassmorphic UI with audio replay & live orb
│
├── models/
│   ├── whisper/                 # faster-whisper cache directory
│   ├── piper/                   # Local Piper ONNX voice models
│   └── silero_vad.onnx          # Silero neural voice activity detector
│
├── audio/                       # Working audio buffers and persistent user/reply WAVs
├── credentials/                 # Local credentials and environment overrides (.env)
├── tests/                       # Comprehensive Pytest test suite (136 unit tests)
│   ├── test_dynamic_language_mirroring.py  # Linguistic marker & Indic discrimination tests
│   ├── test_multilingual.py               # Edge-TTS voice mapping & script tests
│   ├── test_jwt_auth.py                   # JWT security & token lifecycle tests
│   ├── test_server_knowledge_and_chat.py  # RAG & voice chat endpoint tests
│   └── ...                                # Telephony, analytics, and VAD tests
│
├── requirements.txt             # Python dependencies
└── pytest.ini                   # Test runner configuration
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Linux / macOS / Windows (WSL2)**
- **Python 3.12+**
- **Ollama**: Installed and running (`ollama serve`)
  ```bash
  ollama pull gemma3:4b
  ```
- **FFmpeg**: Required for audio format transcoding (`sudo dnf install ffmpeg` or `sudo apt install ffmpeg`).

### 2. Installation
```bash
git clone https://github.com/Prashant7972/multi_voice_agent-.git
cd multi_voice_agent-

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Web Assistant
```bash
python app/main.py --ui
```
Open your browser at **http://localhost:8080**:
1. Click **⚡ Instant Demo / Start Exploring**.
2. Select your language from the dropdown (or keep on **`🌐 Auto-Detect Language`**).
3. Click **Start Session** and speak naturally.
4. Review the conversation transcript, click **`▶ Replay Your Voice`** to check your audio, and inspect real-time latency pills.

### 4. Running in Terminal Mode
```bash
python app/main.py
```

---

## ⚙️ Configuration (`credentials/.env`)

Copy `.env.example` to `credentials/.env` and adjust as needed:

```env
# Server Port
PORT=8080
HOST=0.0.0.0

# LLM Engine & Local Ollama
LLM_ENGINE=ollama
LLM_MODEL=gemma3:4b
OLLAMA_HOST=http://127.0.0.1:11434

# Optional: Google Gemini Cloud Fallback
GEMINI_API_KEY=your_gemini_api_key_here

# STT Configuration
WHISPER_MODEL_NAME=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Twilio Telephony (Optional)
TWILIO_ACCOUNT_SID=ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890
PUBLIC_WEBHOOK_URL=https://your-domain.ngrok-free.app

# Security
JWT_SECRET_KEY=your-secure-random-hmac-sha256-key
ALLOW_DEMO_MODE=true
```

---

## 🧪 Testing

Run the automated test suite verifying all 136 test cases:
```bash
.venv/bin/pytest
```

Output:
```text
============================= 136 passed in 39.59s =============================
```

---

## 🛡️ License

MIT License. Free for personal, research, and commercial use.
