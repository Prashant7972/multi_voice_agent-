import os
import wave
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np
from piper import PiperVoice

from app.config import (
    PIPER_MODEL_PATH,
    PIPER_CONFIG_PATH,
    AUDIO_DIR,
    ALSA_DEVICE,
)


class TTSClient:
    def __init__(
        self,
        model_path: Path = PIPER_MODEL_PATH,
        config_path: Path = PIPER_CONFIG_PATH,
        alsa_device: str = ALSA_DEVICE,
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.alsa_device = alsa_device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper model not found at {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Piper config not found at {self.config_path}")

        print(f"Loading Piper voice: {self.model_path.name}...")
        self.voice = PiperVoice.load(str(self.model_path), config_path=str(self.config_path))
        self.sample_rate = self.voice.config.sample_rate
        print(f"TTS Initialized. Native sample rate: {self.sample_rate}Hz")

    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """
        Synthesize text to a WAV audio file.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty for synthesis.")

        if output_path is None:
            output_path = AUDIO_DIR / "speech_output.wav"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunks = list(self.voice.synthesize(text))
        if not chunks:
            raise RuntimeError("Piper synthesis returned no audio chunks.")

        first_chunk = chunks[0]
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(first_chunk.sample_channels)
            wav_file.setsampwidth(first_chunk.sample_width)
            wav_file.setframerate(first_chunk.sample_rate)
            for chunk in chunks:
                wav_file.writeframes(chunk.audio_int16_bytes)

        return output_path

    def play(self, wav_path: Path) -> None:
        """
        Play audio file through aplay (ALSA with automatic resampling) or sounddevice.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file {wav_path} does not exist.")

        # Try aplay first (handles sample rate conversion smoothly on Linux)
        if shutil.which("aplay"):
            try:
                subprocess.run(
                    ["aplay", "-q", "-D", self.alsa_device, str(wav_path)],
                    check=True,
                    stderr=subprocess.DEVNULL,
                )
                return
            except Exception:
                # If specified device fails, try default aplay
                try:
                    subprocess.run(
                        ["aplay", "-q", str(wav_path)],
                        check=True,
                        stderr=subprocess.DEVNULL,
                    )
                    return
                except Exception:
                    pass

        # Fallback to sounddevice
        try:
            import sounddevice as sd
            with wave.open(str(wav_path), "rb") as wf:
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Audio playback error: {e}")

    def speak(self, text: str, output_path: Optional[Path] = None, play_audio: bool = True) -> Path:
        """
        Synthesize and optionally play speech.
        """
        wav_file = self.synthesize(text, output_path)
        if play_audio:
            self.play(wav_file)
        return wav_file
