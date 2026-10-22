# Local Voice Agent 🎙️

A completely local, free, offline, and private voice assistant running on your machine with zero cloud AI APIs, now featuring **Twilio Telephone Voice Calling** and **Dual Model Switching** (Qwen 2.5 vs. NVIDIA Nemotron).

```text
Microphone / Phone Call  ──▶  Silero VAD  ──▶  faster-whisper  ──▶  Ollama (Qwen / Nemotron)  ──▶  Piper TTS  ──▶  Speaker / Phone Earpiece
```

---

## 🌟 Key Features

* **100% Local & Private AI Core**:
  - LLM runs locally via Ollama (`http://127.0.0.1:11435`).
  - STT (`faster-whisper`) runs on CPU using `int8` CTranslate2 (~150ms latency).
  - TTS (`Piper`) runs on CPU with ONNX Runtime (instant synthesis).
* **Twilio Voice Phone Integration (Inbound & Outbound)**:
  - Call into your local assistant from any standard mobile phone or landline.
  - Initiate outbound calls directly from the Web UI dialer.
  - Real-time bidirectional audio streaming via WebSockets with on-the-fly 8kHz $\mu$-law $\longleftrightarrow$ 16kHz linear PCM transcoding.
* **Dual Model Switcher**:
  - **Qwen 2.5 Coder 3B** (`qwen2.5-coder:3b-instruct`): Default model. Extremely fast (~34 tokens/s, ~0.3s latency), lightweight (~2.1 GB VRAM), optimal for real-time voice and phone conversations.
  - **NVIDIA Nemotron 3 Nano 4B** (`hf.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M`): High reasoning capacity (~2.9 GB VRAM), suitable for complex multi-step reasoning.
  - Dynamically switch models via Web UI dropdown or API without restarting the server!
* **Zero Button Hands-Free Interaction**:
  - Automatically detects speech and triggers answers upon a natural pause.
  - Default silence pause: **1.0 second** (customizable in real-time with the UI slider or `--silence-seconds` CLI flag).
* **Modern Web UI**:
  - Glowing pulsing voice visualizer orb with real-time waveform canvas.
  - Silence progress bar showing live pause countdown.
  - Integrated phone dialer card with live call status.
  - Multi-turn conversation transcript with latency tracking.

---

## 📂 Project Structure

```text
local-voice-agent/
│
├── .venv/                         # Python 3.12 Virtual Environment
├── app/
│   ├── __init__.py
│   ├── config.py                  # Hyperparameters, Ollama port 11435 & Twilio config
│   ├── audio.py                   # ALSA capture, Silero VAD, and playback
│   ├── stt.py                     # faster-whisper STT client
│   ├── llm.py                     # Ollama client with model switching & speech cleaner
│   ├── tts.py                     # Piper TTS client
│   ├── twilio_handler.py          # Twilio WebSocket streaming, transcoding & call dispatcher
│   ├── server.py                  # FastAPI server (Web UI, /api/chat, /twiml/stream, /twiml/incoming)
│   ├── main.py                    # Voice Agent loop and CLI
│   └── templates/
│       └── index.html             # Glowing visualizer UI, dialer & model selector
│
├── models/
│   ├── piper/                     # en_US-lessac-medium ONNX voice
│   ├── whisper/                   # Whisper base model cache
│   └── silero_vad.onnx            # Silero VAD v5 model
│
├── audio/                         # Generated & recorded audio buffers
├── tests/
│   ├── test_llm.py                # LLM test
│   ├── test_tts.py                # TTS test
│   ├── test_stt.py                # STT test
│   ├── test_pipeline.py           # Full loop local test
│   └── test_twilio.py             # Twilio streaming & model switcher test suite
│
├── .env.example                   # Environment configuration template
├── .env                           # Local settings (Ollama port, Twilio credentials)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Virtual Environment
```bash
cd /home/prashant/.gemini/antigravity/scratch/local-voice-agent
source .venv/bin/activate
```

### 2. Run the Automated Tests
```bash
# Verify Twilio streaming, transcoding, and model selection
python tests/test_twilio.py

# Verify LLM connection (Ollama port 11435)
python tests/test_llm.py

# Verify Piper TTS
python tests/test_tts.py
```

---

## 🖥️ Running the Voice Assistant

### Option 1: Web Voice Assistant + Phone Bridge (Recommended)
```bash
python app/main.py --ui
```
👉 Open your browser at **http://localhost:8000**
- Click **Start Hands-Free** to speak into your laptop mic.
- Select your active model (Qwen 3B or Nemotron 4B) in the top header.
- Adjust the silence timeout slider (defaults to 1.0s).

### Option 2: Terminal Hands-Free Mode
```bash
python app/main.py
```
*(Optionally change silence duration: `python app/main.py --silence-seconds 1.0`)*.

---

## 📞 Twilio Voice Setup (Phone Calling)

To allow anyone to call your local assistant from their mobile phone, or to place outbound calls:

### Step 1: Configure `.env`
Add your Twilio credentials to `.env`:
```env
TWILIO_ACCOUNT_SID=ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890
PUBLIC_WEBHOOK_URL=https://your-domain.ngrok-free.app
```
*(Note: If credentials are not yet entered, the UI operates in simulated test mode).*

### Step 2: Expose your Local Port with a Tunnel
Run ngrok or cloudflared to expose port 8000:
```bash
ngrok http 8000
```
Copy the forwarding HTTPS URL (e.g., `https://abc123.ngrok-free.app`) and set it as `PUBLIC_WEBHOOK_URL` in `.env`.

### Step 3: Configure Twilio Inbound Webhook
1. Go to your **Twilio Console** ➔ **Phone Numbers** ➔ **Manage** ➔ **Active numbers**.
2. Select your Twilio phone number.
3. Under **Voice & Fax** ➔ **A Call Comes In**:
   - Select **Webhook**.
   - URL: `https://your-domain.ngrok-free.app/twiml/incoming`
   - Method: **HTTP POST**.
4. Save configuration.

### Step 4: Make & Receive Calls
- **Inbound**: Dial your Twilio phone number from your cell phone. The assistant will answer and talk back in real-time!
- **Outbound**: In the Web UI, expand the **📞 Twilio Phone Bridge** card, enter your phone number, and click **Call**.
