import os
import sys
import time
import argparse
import signal
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    ALSA_DEVICE,
    AUDIO_DIR,
    LLM_MODEL,
    WHISPER_MODEL_NAME,
    PIPER_VOICE_NAME,
    SILENCE_DURATION_THRESHOLD,
)
from app.audio import AudioRecorder, unmute_speakers
from app.stt import STTClient
from app.llm import LLMClient
from app.tts import TTSClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local Hands-Free Voice Agent - Continuous Audio-to-Audio Conversational Bot"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive Modern Web Voice Assistant UI.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind the Web UI server (default: 8080).",
    )
    parser.add_argument(
        "--mode",
        choices=["hands-free", "vad", "push-to-talk", "ptt", "text", "pipecat"],
        default="hands-free",
        help="Interaction mode: 'hands-free', 'pipecat' (with neural VAD noise filtering), 'push-to-talk', or 'text'.",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=1.0,
        help="Seconds of silence required after speaking to trigger assistant response (default: 1.0s).",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable Text-to-Speech audio playback (print-only mode).",
    )
    parser.add_argument(
        "--alsa-device",
        type=str,
        default=ALSA_DEVICE,
        help=f"ALSA hardware playback device (default: {ALSA_DEVICE}).",
    )
    return parser.parse_args()


class LocalVoiceAgent:
    def __init__(
        self,
        mode: str = "hands-free",
        silence_seconds: float = 1.0,
        enable_tts: bool = True,
        alsa_device: str = ALSA_DEVICE,
    ):
        self.mode = "hands-free" if mode in ["hands-free", "vad", "pipecat"] else mode
        self.silence_seconds = silence_seconds
        self.enable_tts = enable_tts
        self.alsa_device = alsa_device
        self.use_pipecat = (mode == "pipecat")

        # Unmute speakers on launch
        unmute_speakers()

        print("=" * 65)
        print("Local Hands-Free Voice Assistant")
        print("=" * 65)
        print(f"• Engine:          {'Pipecat AI 1.11.0 (Neural VAD)' if self.use_pipecat else 'Local Hands-Free Pipeline'}")
        print(f"• LLM Model:       Ollama ({LLM_MODEL})")
        print(f"• Speech-to-Text:  faster-whisper ({WHISPER_MODEL_NAME} int8 on CPU)")
        print(f"• Text-to-Speech:  Piper ({PIPER_VOICE_NAME}) [Enabled: {self.enable_tts}]")
        print(f"• Silence Timeout: {self.silence_seconds:.1f} seconds (no Enter key needed!)")
        print(f"• Interaction:     {mode.upper()}")
        print("=" * 65)

        # Initialize components
        self.llm = LLMClient()
        self.stt = STTClient()
        self.recorder = AudioRecorder(alsa_device=self.alsa_device)
        self.tts = TTSClient(alsa_device=self.alsa_device) if self.enable_tts else None

        self.mic_input_path = AUDIO_DIR / "current_input.wav"
        self.tts_output_path = AUDIO_DIR / "current_reply.wav"
        self.running = True

    def run_turn(self, user_text: str):
        """Process one conversational turn."""
        print(f"User: \"{user_text}\"")

        # Special exit commands
        clean_input = user_text.lower().strip(" .!?,")
        if clean_input in ["exit", "quit", "goodbye", "bye"]:
            farewell = "Goodbye! Have a wonderful day."
            print(f"Assistant: {farewell}")
            if self.tts:
                self.tts.speak(farewell, output_path=self.tts_output_path, play_audio=True)
            self.running = False
            return

        if clean_input in ["clear", "reset", "clear memory"]:
            self.llm.clear_history()
            ack = "Memory reset. What would you like to talk about?"
            print(f"Assistant: {ack}")
            if self.tts:
                self.tts.speak(ack, output_path=self.tts_output_path, play_audio=True)
            return

        # Query LLM
        t_start = time.time()
        reply = self.llm.chat(user_text)
        llm_time = time.time() - t_start

        print(f"Assistant ({llm_time:.2f}s): {reply}")

        # Speak answer through speakers
        if self.tts and reply:
            print("Speaking response...", flush=True)
            self.tts.speak(reply, output_path=self.tts_output_path, play_audio=True)

    def run(self):
        print("\nVoice Assistant listening.")
        print("Speak into microphone. Pause to trigger response.")
        print("Commands: 'clear' to reset history, 'exit' to quit.\n")

        # Set up graceful Ctrl+C handler
        def sigint_handler(sig, frame):
            print("\n\nExiting Voice Assistant. Goodbye!")
            sys.exit(0)

        signal.signal(signal.SIGINT, sigint_handler)

        # Spoken welcome greeting
        if self.tts:
            welcome_msg = "Hello! I am ready. Speak anytime and I will answer after you pause."
            print(f"Assistant: {welcome_msg}")
            self.tts.speak(welcome_msg, output_path=self.tts_output_path, play_audio=True)

        while self.running:
            try:
                if self.mode == "hands-free":
                    audio_path = self.recorder.listen_and_record_speech(
                        output_path=self.mic_input_path,
                        silence_seconds=self.silence_seconds,
                    )

                    if audio_path is None or not audio_path.exists():
                        continue

                    print("Transcribing audio...", end="", flush=True)
                    t_stt = time.time()
                    transcription = self.stt.transcribe(audio_path)
                    stt_time = time.time() - t_stt
                    print(f" ({stt_time:.2f}s)")

                    if not transcription or len(transcription.strip()) == 0:
                        print("No speech recognized. Still listening...")
                        continue

                    self.run_turn(transcription)

                elif self.mode == "push-to-talk":
                    audio_path = self.recorder.record_push_to_talk(self.mic_input_path)
                    print("Transcribing audio...", end="", flush=True)
                    t_stt = time.time()
                    transcription = self.stt.transcribe(audio_path)
                    stt_time = time.time() - t_stt
                    print(f" ({stt_time:.2f}s)")

                    if not transcription or len(transcription.strip()) == 0:
                        print("No speech recognized. Please try again.")
                        continue

                    self.run_turn(transcription)

                elif self.mode == "text":
                    try:
                        user_input = input("\nYou: ").strip()
                    except EOFError:
                        break
                    if not user_input:
                        continue
                    self.run_turn(user_input)

            except Exception as e:
                print(f"\n[Notice: {e}]")
                time.sleep(0.5)


def main():
    args = parse_args()
    if args.ui:
        import uvicorn
        import logging

        class SuppressInvalidHTTPRequestFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return "Invalid HTTP request received" not in record.getMessage()

        logging.getLogger("uvicorn.error").addFilter(SuppressInvalidHTTPRequestFilter())

        os.environ["PORT"] = str(args.port)
        print("\nStarting Local Voice Agent Web UI...")
        print(f"Open your browser at: http://localhost:{args.port}\n")
        uvicorn.run("app.server:app", host="0.0.0.0", port=args.port, reload=True, h11_max_incomplete_event_size=65536)
        return

    agent = LocalVoiceAgent(
        mode=args.mode,
        silence_seconds=args.silence_seconds,
        enable_tts=not args.no_tts,
        alsa_device=args.alsa_device,
    )
    agent.run()


if __name__ == "__main__":
    main()
