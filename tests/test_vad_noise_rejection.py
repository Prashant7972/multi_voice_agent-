import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import wave
import numpy as np
from app.pipecat_agent import PipecatNeuralVAD
from pipecat.audio.vad.vad_analyzer import VADState

async def test_noise_vs_speech():
    print("=" * 60)
    print("🧪 TESTING PIPECAT NEURAL VAD: NOISE REJECTION VS SPEECH")
    print("=" * 60)

    vad = PipecatNeuralVAD(confidence=0.75, start_secs=0.20, min_volume=0.06)

    # 1. Test Artificial Background White Noise
    print("\n1. Testing Background Ambient Noise (Random Hiss / Fan Hum)...")
    noise_interrupted = False
    max_noise_conf = 0.0
    for _ in range(30):
        # Moderate amplitude random noise
        noise_chunk = (np.random.normal(0, 0.08, 512) * 32767).astype(np.int16).tobytes()
        state, conf = await vad.analyze_pcm_chunk(noise_chunk)
        if conf > max_noise_conf:
            max_noise_conf = conf
        if state == VADState.SPEAKING:
            noise_interrupted = True
            break

    print(f"   Max noise confidence score: {max_noise_conf}")
    print(f"   Noise triggered speech: {noise_interrupted}")
    assert not noise_interrupted, "FAIL: Background noise falsely triggered speech interruption!"
    print("   ✅ PASS: Background noise successfully rejected (no false interruption).")

    # 2. Test Real Human Speech File
    print("\n2. Testing Real Human Speech ('audio/test_stage3_what_is_python.wav')...")
    vad = PipecatNeuralVAD(confidence=0.75, start_secs=0.20, min_volume=0.06)
    speech_detected = False
    max_speech_conf = 0.0

    with wave.open("audio/test_stage3_what_is_python.wav", "rb") as wf:
        num_frames = 512
        frame_bytes = num_frames * 2
        while True:
            chunk = wf.readframes(num_frames)
            if len(chunk) < frame_bytes:
                break
            state, conf = await vad.analyze_pcm_chunk(chunk)
            if conf > max_speech_conf:
                max_speech_conf = conf
            if state == VADState.SPEAKING:
                speech_detected = True

    print(f"   Max speech confidence score: {max_speech_conf}")
    print(f"   Human speech confirmed: {speech_detected}")
    assert speech_detected, "FAIL: Human speech failed to be detected!"
    print("   ✅ PASS: Real human speech accurately confirmed (Barge-In triggered).")
    print("\n🎉 ALL PIPECAT NEURAL VAD NOISE TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(test_noise_vs_speech())
