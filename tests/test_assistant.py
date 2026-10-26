import pytest
from app.assistant import VoiceAssistant, get_assistant, DEFAULT_KEEP_ALIVE
from app.config import OLLAMA_KEEP_ALIVE

def test_assistant_default_keep_alive():
    assert DEFAULT_KEEP_ALIVE == "30m"
    assert OLLAMA_KEEP_ALIVE == "30m"

    assistant = VoiceAssistant()
    assert assistant.keep_alive == "30m"
    assert assistant.client.keep_alive == "30m"

def test_get_assistant_singleton():
    a1 = get_assistant()
    a2 = get_assistant()
    assert a1 is a2
    assert a1.keep_alive == "30m"

def test_assistant_chat_mocked():
    assistant = VoiceAssistant()
    # Mock client for offline test environment
    assistant.client.ollama_client.chat = lambda *args, **kwargs: {
        "message": {"content": "Hello! I am ready to help."}
    }
    assistant.client.gemini_client.chat = lambda *args, **kwargs: {
        "message": {"content": "Hello! I am ready to help."}
    }
    reply = assistant.chat("Hello there")
    assert "Hello" in reply

def test_assistant_chat_messages_mocked():
    assistant = VoiceAssistant()
    assistant.client.ollama_client.chat = lambda *args, **kwargs: {
        "message": {"content": "Sure, what can I do for you?"}
    }
    reply = assistant.chat_messages([{"role": "user", "content": "Can you help?"}])
    assert "Sure" in reply

def test_assistant_warmup():
    assistant = VoiceAssistant()
    # Mock warmup chat call
    assistant.client.ollama_client.chat = lambda *args, **kwargs: {
        "message": {"content": "hi"}
    }
    res = assistant.warmup()
    assert res is True
