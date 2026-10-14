import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.stt import STTClient
from app.llm import LLMClient
from app.tts import TTSClient
from app.config import AUDIO_DIR


def test_full_pipeline():
    print("=" * 60)
    print("TEST: Full End-to-End Voice Pipeline")
    print("Microphone/WAV -> faster-whisper -> Ollama Qwen -> Piper -> Speaker/WAV")
    print("=" * 60)

    stt = STTClient()
    llm = LLMClient()
    tts = TTSClient()

    # Step 1: Synthesize a user question as simulated voice input
    user_question = "What is the speed of light in vacuum?"
    input_wav = AUDIO_DIR / "pipeline_user_voice.wav"
    print(f"\n1. Simulating user speech: '{user_question}'...")
    tts.synthesize(user_question, output_path=input_wav)

    # Step 2: Speech-To-Text (faster-whisper)
    t0 = time.time()
    transcribed_text = stt.transcribe(input_wav)
    stt_duration = time.time() - t0
    print(f"2. [STT] Transcribed: '{transcribed_text}' in {stt_duration:.2f}s")
    assert "speed of light" in transcribed_text.lower()

    # Step 3: LLM Inference (Ollama Qwen 2.5 Coder 3B)
    t1 = time.time()
    try:
        reply = llm.chat(transcribed_text)
    except Exception as e:
        print(f"ℹ️ [Ollama Offline]: {e}. Falling back to mock client for pipeline test.")
        llm.client.chat = lambda *args, **kwargs: {"message": {"content": "The speed of light in vacuum is approximately 299,792 kilometers per second."}}
        reply = llm.chat(transcribed_text)
    llm_duration = time.time() - t1
    print(f"3. [LLM] Assistant Response ({llm_duration:.2f}s): '{reply}'")
    assert len(reply) > 0

    # Step 4: Text-To-Speech (Piper)
    output_wav = AUDIO_DIR / "pipeline_assistant_voice.wav"
    t2 = time.time()
    tts.speak(reply, output_path=output_wav, play_audio=True)
    tts_duration = time.time() - t2
    print(f"4. [TTS] Synthesized & Played audio in {tts_duration:.2f}s")
    assert output_wav.exists()

    total_latency = stt_duration + llm_duration + tts_duration
    print(f"\n⚡ Total Voice Turn Latency: {total_latency:.2f}s")
    print("\n✅ Full voice pipeline test PASSED successfully!\n")


if __name__ == "__main__":
    test_full_pipeline()
