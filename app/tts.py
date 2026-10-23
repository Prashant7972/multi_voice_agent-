import asyncio
import concurrent.futures
import logging
import os
import wave
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
from piper import PiperVoice

from app.config import (
    PIPER_MODEL_PATH,
    PIPER_CONFIG_PATH,
    AUDIO_DIR,
    ALSA_DEVICE,
)

logger = logging.getLogger("voice_agent.tts")

LANGUAGE_VOICE_MAP: Dict[str, str] = {
    "en": "en-US-AvaNeural",
    "hi": "hi-IN-MadhurNeural",
    "mr": "mr-IN-AarohiNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-VivienneMultilingualNeural",
    "de": "de-DE-SeraphinaMultilingualNeural",
    "it": "it-IT-DiegoNeural",
    "ja": "ja-JP-KeitaNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ru": "ru-RU-DmitryNeural",
    "pt": "pt-BR-AntonioNeural",
    "ar": "ar-SA-HamedNeural",
    "ta": "ta-IN-ValluvarNeural",
    "te": "te-IN-MohanNeural",
    "bn": "bn-IN-BashkarNeural",
    "gu": "gu-IN-NiranjanNeural",
    "kn": "kn-IN-GaganNeural",
    "ml": "ml-IN-MidhunNeural",
    "pa": "pa-IN-OjasNeural",
    "ur": "ur-PK-AsadNeural",
    "ko": "ko-KR-InJoonNeural",
    "nl": "nl-NL-MaartenNeural",
    "tr": "tr-TR-AhmetNeural",
    "pl": "pl-PL-MarekNeural",
    "sv": "sv-SE-MattiasNeural",
    "id": "id-ID-ArdiNeural",
    "vi": "vi-VN-NamMinhNeural",
    "th": "th-TH-NiwatNeural",
}

LANGUAGE_NAME_ALIASES: Dict[str, str] = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "chinese": "zh",
    "mandarin": "zh",
    "russian": "ru",
    "portuguese": "pt",
    "arabic": "ar",
    "tamil": "ta",
    "telugu": "te",
    "bengali": "bn",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "punjabi": "pa",
    "urdu": "ur",
    "korean": "ko",
    "dutch": "nl",
    "turkish": "tr",
    "polish": "pl",
    "swedish": "sv",
    "indonesian": "id",
    "vietnamese": "vi",
    "thai": "th",
}


def normalize_language_code(lang: Optional[str]) -> str:
    """Normalizes language code or full name to standard 2-letter ISO code."""
    if not lang:
        return "en"
    clean = lang.strip().lower()
    if clean in LANGUAGE_NAME_ALIASES:
        return LANGUAGE_NAME_ALIASES[clean]
    if len(clean) > 2 and clean[:2] in LANGUAGE_VOICE_MAP:
        return clean[:2]
    return clean


def _run_async(coro):
    """Executes an async coroutine synchronously, safe inside or outside running event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


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

        logger.info("Loading Piper voice: %s...", self.model_path.name)
        self.voice = PiperVoice.load(str(self.model_path), config_path=str(self.config_path))
        self.sample_rate = self.voice.config.sample_rate
        logger.info("TTS Initialized. Native sample rate: %dHz", self.sample_rate)

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Returns the list of supported language codes."""
        return sorted(list(LANGUAGE_VOICE_MAP.keys()))

    @classmethod
    def get_voice_for_language(cls, lang: str) -> str:
        """Returns the mapped neural voice name for the given language."""
        code = normalize_language_code(lang)
        return LANGUAGE_VOICE_MAP.get(code, "en-US-AvaNeural")

    def _synthesize_piper(self, text: str, output_path: Path) -> Path:
        """Fallback synthesis using local Piper neural voice."""
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

    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        language: str = "en",
    ) -> Path:
        """
        Synthesize text to a WAV audio file.
        For English ('en'), utilizes local Piper neural voice for lowest offline latency.
        For non-English languages (Hindi, Spanish, French, German, Marathi, Japanese, etc.),
        utilizes Edge-TTS neural multilingual voices, automatically converting to 16kHz WAV.
        Falls back seamlessly to local Piper if network is unreachable or Edge-TTS is unavailable.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty for synthesis.")

        if output_path is None:
            output_path = AUDIO_DIR / "speech_output.wav"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        lang_code = normalize_language_code(language)

        # For non-English languages, attempt Edge-TTS neural speech synthesis
        if lang_code != "en" and lang_code in LANGUAGE_VOICE_MAP:
            temp_mp3 = output_path.with_suffix(f".tmp.{os.getpid()}.mp3")
            try:
                voice_name = LANGUAGE_VOICE_MAP[lang_code]

                async def _generate_edge_audio():
                    import edge_tts
                    communicate = edge_tts.Communicate(text, voice_name)
                    await communicate.save(str(temp_mp3))

                _run_async(_generate_edge_audio())

                if temp_mp3.exists() and temp_mp3.stat().st_size > 0:
                    if shutil.which("ffmpeg"):
                        subprocess.run(
                            [
                                "ffmpeg", "-y", "-i", str(temp_mp3),
                                "-ar", "16000", "-ac", "1",
                                str(output_path),
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        if temp_mp3.exists():
                            try:
                                temp_mp3.unlink()
                            except Exception:
                                pass
                        return output_path
            except Exception as e:
                logger.warning(
                    "Edge-TTS synthesis failed for lang '%s' (%s). Falling back to local Piper TTS.",
                    language,
                    e,
                )
            finally:
                if temp_mp3.exists():
                    try:
                        temp_mp3.unlink()
                    except Exception:
                        pass

        # Native Piper Local Voice Synthesis (English & Offline Fallback)
        return self._synthesize_piper(text, output_path)

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
            logger.warning("Audio playback error: %s", e)

    def speak(
        self,
        text: str,
        output_path: Optional[Path] = None,
        play_audio: bool = True,
        language: str = "en",
    ) -> Path:
        """
        Synthesize and optionally play speech.
        """
        wav_file = self.synthesize(text, output_path, language=language)
        if play_audio:
            self.play(wav_file)
        return wav_file
