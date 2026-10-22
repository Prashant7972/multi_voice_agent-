import asyncio
import io
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService

logger = logging.getLogger("voice_agent.pipecat")

from app.config import (
    BASE_DIR,
    OLLAMA_HOST,
    LLM_MODEL,
    SYSTEM_PROMPT,
    WHISPER_MODEL_NAME,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CACHE_DIR,
    PIPER_DIR,
    PIPER_MODEL_PATH,
    PIPER_VOICE_NAME,
)


class PipecatNeuralVAD:
    """
    Dedicated Neural Voice Activity Detector powered by Pipecat's Silero VAD.
    Rejects background noise, fan hum, keyboard clicks, breathing, and room reverberation.
    Only triggers when human vocalization confidence >= confidence_threshold (default: 0.75).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        confidence: float = 0.75,
        start_secs: float = 0.20,
        stop_secs: float = 0.80,
        min_volume: float = 0.06,
    ):
        self.sample_rate = sample_rate
        self.params = VADParams(
            confidence=confidence,
            start_secs=start_secs,
            stop_secs=stop_secs,
            min_volume=min_volume,
        )
        self.vad = SileroVADAnalyzer(sample_rate=sample_rate, params=self.params)
        self.vad.set_sample_rate(sample_rate)

    async def analyze_pcm_chunk(self, pcm_bytes: bytes) -> Tuple[VADState, float]:
        """
        Analyzes a raw PCM audio chunk (16-bit signed integer, mono).
        Returns (VADState, confidence_score as float).
        """
        if not pcm_bytes:
            return VADState.QUIET, 0.0

        raw_conf = self.vad.voice_confidence(pcm_bytes)
        try:
            confidence = float(np.squeeze(raw_conf))
        except Exception:
            confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.0

        state = await self.vad.analyze_audio(pcm_bytes)
        return state, round(confidence, 4)

    def is_speech_confident(self, confidence: float) -> bool:
        """Determines if the confidence score meets the human speech requirement."""
        return confidence >= self.params.confidence


class PipecatVoiceAgent:
    """
    Pipecat Conversational Voice Agent orchestrating:
    - Silero Neural VAD (background noise filtering & turn-taking)
    - Whisper STT (faster-whisper int8 CPU)
    - Ollama LLM (local Qwen 2.5 on GPU)
    - Piper TTS (local neural speech synthesis)
    """

    def __init__(
        self,
        ollama_host: str = OLLAMA_HOST,
        model_name: str = LLM_MODEL,
        whisper_model: str = WHISPER_MODEL_NAME,
        piper_voice: str = PIPER_VOICE_NAME,
    ):
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.whisper_model = whisper_model
        self.piper_voice = piper_voice

        # 1. Neural VAD with strict noise filtering
        self.vad = PipecatNeuralVAD(
            sample_rate=16000,
            confidence=0.75,
            start_secs=0.20,
            stop_secs=0.80,
            min_volume=0.06,
        )

        # 2. Local Whisper STT Service
        logger.info("Initializing Whisper STT Service...")
        try:
            self.stt = WhisperSTTService(
                settings=WhisperSTTService.Settings(model=self.whisper_model),
                device=WHISPER_DEVICE if WHISPER_DEVICE == "cpu" else "cpu",
                compute_type=WHISPER_COMPUTE_TYPE,
            )
        except Exception as e:
            logger.warning("Pipecat STT Warning: %s. Using stub for offline / CI runner.", e)
            class StubSTTService:
                def __init__(self):
                    pass
            self.stt = StubSTTService()

        # 3. Local Ollama LLM Service
        base_url = f"{self.ollama_host.rstrip('/')}/v1"
        logger.info("Initializing Ollama LLM Service (%s at %s)...", self.model_name, base_url)
        self.llm = OLLamaLLMService(
            settings=OLLamaLLMService.Settings(model=self.model_name),
            base_url=base_url,
        )

        # 4. Local Piper TTS Service
        logger.info("Initializing Piper TTS Service (%s)...", self.piper_voice)
        try:
            self.tts = PiperTTSService(
                download_dir=PIPER_DIR,
                settings=PiperTTSService.Settings(voice=self.piper_voice),
            )
        except Exception as e:
            logger.warning("Pipecat TTS Warning: %s. Using stub for offline / CI runner.", e)
            class StubTTSService:
                def __init__(self):
                    pass
            self.tts = StubTTSService()

        logger.info("Pipecat Conversational Pipeline initialized.")

    def get_vad(self) -> PipecatNeuralVAD:
        return self.vad
