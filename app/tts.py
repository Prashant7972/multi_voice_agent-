import asyncio
import concurrent.futures
import logging
import os
import wave
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np
from piper import PiperVoice

from app.config import (
    PIPER_MODEL_PATH,
    PIPER_CONFIG_PATH,
    AUDIO_DIR,
    ALSA_DEVICE,
)

logger = logging.getLogger("voice_agent.tts")

VOICE_DEMOGRAPHICS: Dict[str, List[Dict[str, str]]] = {
    "genders": [
        {"id": "female", "label": "Female", "icon": "👩", "desc": "Natural feminine inflection & tone"},
        {"id": "male", "label": "Male", "icon": "👨", "desc": "Resonant masculine resonance & timbre"},
        {"id": "child", "label": "Child", "icon": "🧒", "desc": "High-spirited, youthful cadence (8–14 yrs)"},
        {"id": "aged", "label": "Senior / Aged", "icon": "👴", "desc": "Distinguished, warm elder wisdom (60+ yrs)"},
    ],
    "age_groups": [
        {"id": "child", "label": "Child (8–14 yrs)", "icon": "🎈"},
        {"id": "young_adult", "label": "Young Adult (18–28 yrs)", "icon": "⚡"},
        {"id": "mature_adult", "label": "Mature Adult (30–55 yrs)", "icon": "💼"},
        {"id": "senior", "label": "Senior / Aged (60+ yrs)", "icon": "👓"},
    ],
    "personalities": [
        {"id": "friendly_warm", "label": "Friendly & Warm", "desc": "Approachable, conversational, inviting"},
        {"id": "professional_crisp", "label": "Professional & Crisp", "desc": "Executive, articulate, goal-oriented"},
        {"id": "empathetic_caregiver", "label": "Empathetic Caregiver", "desc": "Gentle, supportive, active listener"},
        {"id": "energetic_sales", "label": "Energetic Sales", "desc": "Upbeat, persuasive, high-momentum"},
        {"id": "calm_advisor", "label": "Calm Advisor", "desc": "Reassuring, thoughtful, grounded"},
    ],
}

VOICE_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "en-US-AvaNeural",
        "name": "Ava",
        "display_name": "Ava (Warm & Welcoming)",
        "language": "en",
        "language_name": "English (US)",
        "gender": "female",
        "age_group": "young_adult",
        "age_label": "Young Adult (18-28)",
        "personality": "friendly_warm",
        "personality_label": "Friendly & Warm",
        "description": "Expressive, welcoming natural tone suited for customer engagement and concierge assistants.",
        "pitch_default": "+0Hz",
        "rate_default": "+0%",
        "sample_phrase": "Hello! I am Ava, your AI voice assistant. How can I help you today?",
    },
    {
        "id": "en-US-JennyNeural",
        "name": "Jenny",
        "display_name": "Jenny (Executive & Crisp)",
        "language": "en",
        "language_name": "English (US)",
        "gender": "female",
        "age_group": "mature_adult",
        "age_label": "Mature Adult (30-55)",
        "personality": "professional_crisp",
        "personality_label": "Professional & Crisp",
        "description": "Clear, polished, and articulate speech ideal for executive briefings and business consultations.",
        "pitch_default": "+0Hz",
        "rate_default": "+0%",
        "sample_phrase": "Good day. I am Jenny. Let us review the key objectives and metrics for your organization.",
    },
    {
        "id": "en-US-EmmaNeural",
        "name": "Emma",
        "display_name": "Emma (Empathetic Caregiver)",
        "language": "en",
        "language_name": "English (US)",
        "gender": "female",
        "age_group": "mature_adult",
        "age_label": "Mature Adult (30-55)",
        "personality": "empathetic_caregiver",
        "personality_label": "Empathetic Caregiver",
        "description": "Gentle, compassionate, and attentive voice designed for healthcare, wellness, and support.",
        "pitch_default": "+0Hz",
        "rate_default": "-5%",
        "sample_phrase": "Hello there, I'm Emma. Take a deep breath—I am right here to support you every step of the way.",
    },
    {
        "id": "en-US-AndrewNeural",
        "name": "Andrew",
        "display_name": "Andrew (Dynamic & Confident)",
        "language": "en",
        "language_name": "English (US)",
        "gender": "male",
        "age_group": "young_adult",
        "age_label": "Young Adult (18-28)",
        "personality": "energetic_sales",
        "personality_label": "Energetic Sales",
        "description": "Punchy, persuasive, high-energy cadence perfect for marketing outreach and sales pitches.",
        "pitch_default": "+0Hz",
        "rate_default": "+5%",
        "sample_phrase": "Hey there! Andrew here. We have an exciting opportunity that can instantly accelerate your growth!",
    },
    {
        "id": "en-US-GuyNeural",
        "name": "Guy",
        "display_name": "Guy (Deep & Trustworthy)",
        "language": "en",
        "language_name": "English (US)",
        "gender": "male",
        "age_group": "mature_adult",
        "age_label": "Mature Adult (30-55)",
        "personality": "calm_advisor",
        "personality_label": "Calm Advisor",
        "description": "Deep, reassuring baritone providing stability, strategic guidance, and technical credibility.",
        "pitch_default": "-5Hz",
        "rate_default": "+0%",
        "sample_phrase": "Welcome. I am Guy. Let's analyze the technical architecture and find the most robust solution.",
    },
    {
        "id": "en-US-AnaNeural",
        "name": "Ana",
        "display_name": "Ana (Youthful & Cheerful)",
        "language": "en",
        "language_name": "English (US)",
        "gender": "child",
        "age_group": "child",
        "age_label": "Child (8-14)",
        "personality": "friendly_warm",
        "personality_label": "Friendly & Cheerful",
        "description": "Bright, playful, upbeat voice tailored for educational games, kids tutoring, and family storytelling.",
        "pitch_default": "+15Hz",
        "rate_default": "+5%",
        "sample_phrase": "Hi! I'm Ana! Are you ready to explore some fun science facts and solve new puzzles together?",
    },
    {
        "id": "en-US-BrianNeural",
        "name": "Brian",
        "display_name": "Brian (Distinguished Elder)",
        "language": "en",
        "language_name": "English (US)",
        "gender": "aged",
        "age_group": "senior",
        "age_label": "Senior / Aged (60+)",
        "personality": "calm_advisor",
        "personality_label": "Calm & Wise Advisor",
        "description": "Dignified, warm, seasoned senior voice offering thoughtful mentorship and trusted counsel.",
        "pitch_default": "-10Hz",
        "rate_default": "-5%",
        "sample_phrase": "Greetings, my friend. I am Brian. Through decades of experience, patience and clarity always win the day.",
    },
    {
        "id": "hi-IN-SwaraNeural",
        "name": "Swara",
        "display_name": "Swara (Hindi - Polite & Expressive)",
        "language": "hi",
        "language_name": "Hindi (India)",
        "gender": "female",
        "age_group": "young_adult",
        "age_label": "Young Adult (18-28)",
        "personality": "friendly_warm",
        "personality_label": "Friendly & Warm",
        "description": "मधुर और आत्मीय हिंदी स्वर, ग्राहक सहायता एवं बातचीत के लिए सर्वोत्तम।",
        "pitch_default": "+0Hz",
        "rate_default": "+0%",
        "sample_phrase": "नमस्ते! मैं स्वरा हूँ, आपकी एआई वॉयस साथी। आज मैं आपकी क्या सहायता कर सकती हूँ?",
    },
    {
        "id": "hi-IN-MadhurNeural",
        "name": "Madhur",
        "display_name": "Madhur (Hindi - Clear & Authoritative)",
        "language": "hi",
        "language_name": "Hindi (India)",
        "gender": "male",
        "age_group": "mature_adult",
        "age_label": "Mature Adult (30-55)",
        "personality": "professional_crisp",
        "personality_label": "Professional & Crisp",
        "description": "स्पष्ट एवं पेशेवर पुरुष स्वर, व्यापार एवं सूचना प्रसारण हेतु आदर्श।",
        "pitch_default": "-5Hz",
        "rate_default": "+0%",
        "sample_phrase": "नमस्कार। मैं मधुर हूँ। आपके व्यापारिक अभियानों और रणनीतियों को आगे बढ़ाने के लिए तैयार हूँ।",
    },
    {
        "id": "mr-IN-AarohiNeural",
        "name": "Aarohi",
        "display_name": "Aarohi (Marathi - Melodious & Polite)",
        "language": "mr",
        "language_name": "Marathi (India)",
        "gender": "female",
        "age_group": "young_adult",
        "age_label": "Young Adult (18-28)",
        "personality": "friendly_warm",
        "personality_label": "Friendly & Warm",
        "description": "मराठीतील सुस्पष्ट आणि आल्हाददायक आवाज, संवाद आणि समर्थनासाठी उत्तम।",
        "pitch_default": "+0Hz",
        "rate_default": "+0%",
        "sample_phrase": "नमस्कार! मी आरोही. मी तुम्हाला आज कशी मदत करू शकते?",
    },
    {
        "id": "mr-IN-ManoharNeural",
        "name": "Manohar",
        "display_name": "Manohar (Marathi - Dignified & Crisp)",
        "language": "mr",
        "language_name": "Marathi (India)",
        "gender": "male",
        "age_group": "mature_adult",
        "age_label": "Mature Adult (30-55)",
        "personality": "professional_crisp",
        "personality_label": "Professional & Crisp",
        "description": "गंभीर आणि विश्वासार्ह मराठी आवाज, अधिकृत घोषणा आणि माहितीसाठी उपयुक्त।",
        "pitch_default": "-5Hz",
        "rate_default": "+0%",
        "sample_phrase": "नमस्कार. मी मनोहर. आपल्या सेवेसाठी मी सदैव तत्पर आहे.",
    },
]


def format_pitch(pitch: Any) -> str:
    """Format pitch into standard Edge-TTS pitch string (e.g. '+0Hz', '+15Hz', '-10Hz')."""
    if pitch is None:
        return "+0Hz"
    if isinstance(pitch, (int, float)):
        val = int(round(pitch))
        return f"+{val}Hz" if val >= 0 else f"{val}Hz"
    s = str(pitch).strip()
    if not s or s.lower() == "default":
        return "+0Hz"
    if s.endswith("Hz") or s.endswith("hz") or s.endswith("%"):
        if not s.startswith("+") and not s.startswith("-"):
            return f"+{s}"
        return s
    try:
        val = int(round(float(s)))
        return f"+{val}Hz" if val >= 0 else f"{val}Hz"
    except Exception:
        return "+0Hz"


def format_rate(rate: Any) -> str:
    """Format rate into standard Edge-TTS rate string (e.g. '+0%', '+15%', '-10%')."""
    if rate is None:
        return "+0%"
    if isinstance(rate, (int, float)):
        # Multiplier style (e.g. 0.8 to 1.5)
        if 0.2 <= rate <= 2.5:
            pct = int(round((rate - 1.0) * 100))
            return f"+{pct}%" if pct >= 0 else f"{pct}%"
        val = int(round(rate))
        return f"+{val}%" if val >= 0 else f"{val}%"
    s = str(rate).strip()
    if not s or s.lower() == "default":
        return "+0%"
    if s.endswith("x") or s.endswith("X"):
        try:
            mult = float(s[:-1])
            pct = int(round((mult - 1.0) * 100))
            return f"+{pct}%" if pct >= 0 else f"{pct}%"
        except Exception:
            return "+0%"
    if s.endswith("%"):
        if not s.startswith("+") and not s.startswith("-"):
            return f"+{s}"
        return s
    try:
        val = float(s)
        if 0.2 <= val <= 2.5:
            pct = int(round((val - 1.0) * 100))
            return f"+{pct}%" if pct >= 0 else f"{pct}%"
        ipct = int(round(val))
        return f"+{ipct}%" if ipct >= 0 else f"{ipct}%"
    except Exception:
        return "+0%"

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
        voice_model: Optional[str] = None,
        pitch: Optional[Union[str, int, float]] = None,
        rate: Optional[Union[str, int, float]] = None,
    ) -> Path:
        """
        Synthesize text to a WAV audio file.
        Supports explicit voice_model (e.g. from VOICE_CATALOG or edge_tts) with customized
        pitch and speech rate.
        If voice_model is not explicitly set:
          - For English ('en'), utilizes local Piper neural voice for lowest offline latency.
          - For non-English languages, utilizes mapped Edge-TTS neural multilingual voices.
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
        selected_voice = voice_model or (LANGUAGE_VOICE_MAP.get(lang_code) if lang_code != "en" else None)

        pitch_str = format_pitch(pitch)
        rate_str = format_rate(rate)

        # If a specific neural voice is requested or non-English language mapped
        if selected_voice:
            temp_mp3 = output_path.with_suffix(f".tmp.{os.getpid()}.mp3")
            try:
                async def _generate_edge_audio():
                    import edge_tts
                    communicate = edge_tts.Communicate(
                        text,
                        selected_voice,
                        pitch=pitch_str,
                        rate=rate_str,
                    )
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
                    "Edge-TTS synthesis failed for voice '%s' (%s). Falling back to local Piper TTS.",
                    selected_voice,
                    e,
                )
            finally:
                if temp_mp3.exists():
                    try:
                        temp_mp3.unlink()
                    except Exception:
                        pass

        # Native Piper Local Voice Synthesis (English default & Offline Fallback)
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
        voice_model: Optional[str] = None,
        pitch: Optional[Union[str, int, float]] = None,
        rate: Optional[Union[str, int, float]] = None,
    ) -> Path:
        """
        Synthesize and optionally play speech.
        """
        wav_file = self.synthesize(
            text,
            output_path,
            language=language,
            voice_model=voice_model,
            pitch=pitch,
            rate=rate,
        )
        if play_audio:
            self.play(wav_file)
        return wav_file
