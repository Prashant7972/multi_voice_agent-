import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from app.pipecat_agent import PipecatVoiceAgent, PipecatNeuralVAD
from pipecat.audio.vad.vad_analyzer import VADState

def test_pipecat_init():
    print("=" * 60)
    print("🧪 TESTING PIPECAT VOICE AGENT INITIALIZATION & SERVICES")
    print("=" * 60)
    agent = PipecatVoiceAgent()
    print(f"• VAD Instance: {agent.vad}")
    print(f"• STT Service:  {agent.stt}")
    print(f"• LLM Service:  {agent.llm}")
    print(f"• TTS Service:  {agent.tts}")
    assert agent.vad is not None
    assert agent.stt is not None
    assert agent.llm is not None
    assert agent.tts is not None
    print("✅ PASS: Pipecat Conversational Voice Agent successfully assembled!")

if __name__ == "__main__":
    test_pipecat_init()
