import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import LLMClient


def test_llm():
    print("=" * 50)
    print("STAGE 1: Testing Local LLM (Ollama / Qwen 2.5 Coder 3B)")
    print("=" * 50)

    llm = LLMClient()

    # Test 1: Simple greeting
    prompt1 = "Hello"
    print(f"\nUser: {prompt1}")
    reply1 = llm.chat(prompt1)
    print(f"Assistant: {reply1}")
    assert len(reply1) > 0, "LLM reply 1 should not be empty"

    # Test 2: Concise explanation
    prompt2 = "Explain Python in one sentence."
    print(f"\nUser: {prompt2}")
    reply2 = llm.chat(prompt2)
    print(f"Assistant: {reply2}")
    assert len(reply2) > 0, "LLM reply 2 should not be empty"

    print("\n✅ Stage 1 passed successfully! Text -> LLM -> Text is working.\n")


if __name__ == "__main__":
    test_llm()
