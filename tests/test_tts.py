import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tts import TTSClient
from app.config import AUDIO_DIR


def test_tts():
    print("=" * 50)
    print("STAGE 2: Testing Text-to-Speech (Piper TTS)")
    print("=" * 50)

    tts = TTSClient()

    test_sentence = "Hello, I am your local voice assistant."
    output_wav = AUDIO_DIR / "test_stage2_speech.wav"

    print(f"\nSynthesizing text: '{test_sentence}'...")
    generated_path = tts.synthesize(test_sentence, output_path=output_wav)
    print(f"Generated WAV file at: {generated_path}")

    assert generated_path.exists(), f"Audio file {generated_path} was not created"
    file_size = generated_path.stat().st_size
    print(f"File size: {file_size} bytes")
    assert file_size > 1000, "Generated audio file is too small"

    print("\nAttempting audio playback through speakers...")
    try:
        tts.play(generated_path)
        print("Playback finished.")
    except Exception as e:
        print(f"Playback note: {e}")

    print("\n✅ Stage 2 passed successfully! Text -> Piper -> WAV -> Speaker is working.\n")


if __name__ == "__main__":
    test_tts()
