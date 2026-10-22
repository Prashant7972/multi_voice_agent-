# Local Voice Agent - Project Architecture & Roadmap

## 1. Overview
A 100% private, free, offline local voice assistant engineered specifically for modest consumer hardware:
* **CPU**: AMD Ryzen 5 5600H (6 cores, 12 threads)
* **RAM**: 16 GB
* **GPU**: NVIDIA GeForce GTX 1650 (4 GB VRAM)

Zero cloud dependencies or API keys (`OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY` are not required).

---

## 2. Architecture Diagram

```text
                             ┌───────────────────┐
                             │    Microphone     │
                             └─────────┬─────────┘
                                       │
                                       ▼
                             ┌───────────────────┐
                             │    Silero VAD     │
                             │ (Speech Detection)│
                             └─────────┬─────────┘
                                       │ Audio (WAV 16kHz)
                                       ▼
                             ┌───────────────────┐
                             │  faster-whisper   │
                             │ (base int8 on CPU)│
                             └─────────┬─────────┘
                                       │ Transcribed Text
                                       ▼
                             ┌───────────────────┐
                             │Conversation Memory│
                             │ (Multi-turn Context)│
                             └─────────┬─────────┘
                                       │ Prompt + History
                                       ▼
                             ┌───────────────────┐
                             │   Ollama (LLM)    │
                             │ Qwen2.5-Coder 3B  │
                             └─────────┬─────────┘
                                       │ Clean Text Reply
                                       ▼
                             ┌───────────────────┐
                             │     Piper TTS     │
                             │  (lessac-medium)  │
                             └─────────┬─────────┘
                                       │ Synthesized Audio
                                       ▼
                             ┌───────────────────┐
                             │  Speaker (ALSA)   │
                             │   (plughw:2,0)    │
                             └───────────────────┘
```

---

## 3. Milestones & Implementation Stages

- [x] **Stage 1 — LLM Validation**: Ollama with `qwen2.5-coder:3b-instruct` verified. Tested greeting and single-sentence explanations (`tests/test_llm.py`).
- [x] **Stage 2 — Text-to-Speech**: Piper TTS integrated with `en_US-lessac-medium.onnx`. Tested speech synthesis and ALSA audio playback (`tests/test_tts.py`).
- [x] **Stage 3 — Speech-to-Text**: `faster-whisper` (base int8) integrated. Verified reference speech transcription and microphone audio capture (`tests/test_stt.py`).
- [x] **Stage 4 — STT + LLM Pipeline**: Speech input transcribes and feeds into Ollama to print conversational responses without TTS (`--no-tts` mode).
- [x] **Stage 5 — Full Voice Loop**: Complete audio loop (Mic -> STT -> Ollama -> Piper -> Speaker) implemented and tested.
- [x] **Stage 6 — Conversation Memory**: Multi-turn history tracking tested ("My name is Alex" -> "What is my name?" -> "Your name is Alex").
- [x] **Stage 7 — Voice Activity Detection (VAD)**: Hands-free Silero VAD v5 ONNX integration with 576-sample rolling context window for automatic speech and silence detection.
- [x] **Stage 8 — Low Latency & Voice UX**: Conversational system prompt tuning, markdown stripper for clean speech, push-to-talk, VAD, and text modes.

---

## 4. Hardware Resource Allocation

| Component | Model | Engine | Compute Target | Memory / VRAM Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **LLM** | Qwen 2.5 Coder 3B | Ollama | GPU (CUDA) | ~1.9 GB VRAM |
| **STT** | Whisper Base | CTranslate2 | CPU (`int8`) | ~150 MB RAM |
| **VAD** | Silero VAD v5 | ONNX Runtime | CPU | ~2.3 MB RAM |
| **TTS** | Lessac Medium | Piper ONNX | CPU | ~61 MB RAM |

Total VRAM used is ~2.0 GB out of 4.0 GB available on the GTX 1650, leaving plenty of headroom.

---

## 5. Future Roadmap
- [ ] **Stage 9**: Interruption / Barge-in during TTS playback.
- [ ] **Stage 10**: Controlled Python Function Calling & Tools (`get_time`, `calculate`, `open_application`).
- [ ] **Stage 11**: Secure local desktop automation.


## resolved changes 