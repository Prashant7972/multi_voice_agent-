import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.stt import STTClient
from app.tts import TTSClient
from app.audio import AudioRecorder
from app.config import AUDIO_DIR


def test_stt():
    print("=" * 50)
    print("STAGE 3: Testing Speech-to-Text (faster-whisper)")
    print("=" * 50)

    # Initialize STT
    stt = STTClient()

    # Part 1: Verify STT with a known test sentence synthesized by Piper
    tts = TTSClient()
    test_phrase = "What is Python?"
    test_wav = AUDIO_DIR / "test_stage3_what_is_python.wav"

    print(f"\n1. Generating reference audio: '{test_phrase}'...")
    tts.synthesize(test_phrase, output_path=test_wav)

    print(f"2. Transcribing reference audio with faster-whisper...")
    transcription = stt.transcribe(test_wav)
    print(f"Transcribed Text: '{transcription}'")

    assert "what is python" in transcription.lower(), f"Expected 'what is python', got '{transcription}'"
    print("✅ Reference transcription matches expected text!")

    # Part 2: Verify recording from microphone
    print("\n3. Testing live microphone recording (1.5 seconds)...")
    recorder = AudioRecorder()
    mic_wav = AUDIO_DIR / "test_stage3_mic.wav"
    try:
        recorder.record_fixed_duration(mic_wav, duration=1.5)
        print(f"Microphone test audio recorded to: {mic_wav}")
        mic_text = stt.transcribe(mic_wav)
        print(f"Microphone transcription: '{mic_text}' (ambient/speech)")
    except Exception as e:
        print(f"Microphone recording note: {e}")

    print("\n✅ Stage 3 passed successfully! Audio -> faster-whisper -> Text is working.\n")


if __name__ == "__main__":
    test_stt()
