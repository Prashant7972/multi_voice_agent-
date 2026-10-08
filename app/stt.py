import os
from pathlib import Path
from typing import Union, Optional
import numpy as np
from faster_whisper import WhisperModel

from app.config import (
    WHISPER_MODEL_NAME,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CACHE_DIR,
)


class STTClient:
    def __init__(
        self,
        model_name: str = WHISPER_MODEL_NAME,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        cache_dir: Path = WHISPER_CACHE_DIR,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading faster-whisper '{self.model_name}' on {self.device} ({self.compute_type})...")
        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.cache_dir),
            )
        except Exception as e:
            if self.device != "cpu":
                print(f"Failed to load on {self.device} ({e}), falling back to CPU...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.cache_dir),
                )
            else:
                raise e

        print(f"STT Initialized successfully ({self.device}).")

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = "en",
        beam_size: int = 1,
    ) -> str:
        """
        Transcribes an audio file or numpy float32 audio array into text.
        beam_size=1 uses greedy decoding for lowest latency.
        """
        if isinstance(audio, Path):
            audio = str(audio)

        segments, info = self.model.transcribe(
            audio,
            beam_size=beam_size,
            language=language,
            vad_filter=True,  # filters non-speech segments
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        transcription = " ".join([segment.text.strip() for segment in segments]).strip()
        return transcription
