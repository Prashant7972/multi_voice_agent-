"""
Assistant Module for Voicing AI.
Provides low-latency conversational LLM integration with automatic model keep-alive
and instant warmup to eliminate cold-start reload latency.
"""
import logging
from typing import Optional, Dict, Any, List

from app.config import (
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    LLM_MODEL,
    SYSTEM_PROMPT,
    GEMINI_API_KEY,
)
from app.llm import LLMClient, GeminiApiClient, clean_text_for_speech

logger = logging.getLogger("voice_agent.assistant")

# Default keep-alive configuration (30 minutes to prevent model unloads between turns)
DEFAULT_KEEP_ALIVE: str = "30m"


class VoiceAssistant:
    """
    Dedicated high-performance Voice Assistant wrapper.
    Ensures model weights remain hot in memory (keep_alive="30m")
    and operates with low-latency context windows.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        host: str = OLLAMA_HOST,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        api_key: str = GEMINI_API_KEY,
    ):
        self.keep_alive = keep_alive or DEFAULT_KEEP_ALIVE
        self.client = LLMClient(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            host=host,
            keep_alive=self.keep_alive,
        )

    def warmup(self) -> bool:
        """Warms up the model in RAM/VRAM to eliminate first-turn latency."""
        return self.client.warmup()

    def chat(
        self,
        user_text: str,
        max_tokens: Optional[int] = 45,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        stateless: bool = False,
    ) -> str:
        """Sends a user query and returns a concise, speech-optimized response."""
        return self.client.chat(
            user_text=user_text,
            max_tokens=max_tokens,
            history=history,
            system_prompt=system_prompt,
            stateless=stateless,
        )

    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 45,
        clean_speech: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """Sends multi-turn chat messages with keep_alive='30m' and low latency."""
        return self.client.chat_messages(
            messages=messages,
            max_tokens=max_tokens,
            clean_speech=clean_speech,
            temperature=temperature,
        )


_assistant_instance: Optional[VoiceAssistant] = None


def get_assistant(
    model: str = LLM_MODEL,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
) -> VoiceAssistant:
    """Returns or creates the singleton VoiceAssistant instance."""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = VoiceAssistant(model=model, keep_alive=keep_alive)
    return _assistant_instance


__all__ = [
    "VoiceAssistant",
    "get_assistant",
    "DEFAULT_KEEP_ALIVE",
    "LLMClient",
    "GeminiApiClient",
    "clean_text_for_speech",
]
