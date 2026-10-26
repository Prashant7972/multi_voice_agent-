import logging
import os
import re
from pathlib import Path
from typing import Union, Optional
import numpy as np
from faster_whisper import WhisperModel

from app.config import (
    WHISPER_MODEL_NAME,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CACHE_DIR,
    WHISPER_CPU_THREADS,
)

logger = logging.getLogger("voice_agent.stt")


# Script-conditioning prompts for Whisper to guarantee transcription in native scripts
# rather than phonetically transliterating into Latin/English letters
LANGUAGE_INITIAL_PROMPTS = {
    "mr": "नमस्कार, तुम्ही कसे आहात? हे मराठी संभाषण आहे. मला AI, Weather, तंत्रज्ञान आणि इतर माहिती सांगा.",
    "hi": "नमस्ते, आप कैसे हैं? यह हिंदी वार्तालाप है। मुझे AI, Weather, तकनीक और अन्य जानकारी बताइए।",
    "es": "Hola, ¿cómo estás? Esta es una conversación en español.",
    "fr": "Bonjour, comment allez-vous? C'est une conversation en français.",
    "de": "Hallo, wie geht es dir? Dies ist ein deutsches Gespräch.",
    "it": "Ciao, come stai? Questa è una conversazione in italiano.",
    "ja": "こんにちは、お元気ですか？日本語の会話です。",
    "zh": "你好，这是一段中文对话。",
    "ru": "Здравствуйте, как ваши дела? Это разговор на русском языке.",
    "pt": "Olá, como você está? Esta é uma conversa em português.",
    "ar": "مرحبا، كيف حالك؟ هذه محادثة باللغة العربية.",
    "ta": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்? இது தமிழ் உரையாடல்.",
    "te": "నమస్కారం, మీరు ఎలా ఉన్నారు? ఇది తెలుగు సంభాషణ.",
    "bn": "নমস্কার, আপনি কেমন আছেন? এটি বাংলা কথোপকথন।",
    "gu": "નમસ્તે, તમે કેમ છો? આ ગુજરાતી વાર્તાલાપ છે.",
    "kn": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ? ಇದು ಕನ್ನಡ ಸಂಭಾಷಣೆ.",
    "ml": "നമസ്കാരം, സുഖമാണോ? ഇത് മലയാളം സംഭാഷണമാണ്.",
    "pa": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ? ਇਹ ਪੰਜਾਬੀ ਗੱਲਬਾਤ ਹੈ।",
    "ur": "السلام علیکم، آپ کیسے हैं؟ یہ اردو گفتگو ہے۔",
    "ko": "안녕하세요, 잘 지내시죠? 한국어 대화입니다.",
    "nl": "Hallo, hoe gaat het met je? Dit is een gesprek in het Nederlands.",
    "tr": "Merhaba, nasılsınız? Bu Türkçe bir konuşmadır.",
    "pl": "Cześć, jak się masz? To jest rozmowa po polsku.",
    "sv": "Hej, hur mår du? Detta är ett samtal på svenska.",
    "id": "Halo, apa kabar? Ini adalah percakapan dalam bahasa Indonesia.",
    "vi": "Xin chào, bạn khỏe không? Đây là cuộc trò chuyện bằng tiếng Việt.",
    "th": "สวัสดีครับ คุณสบายดีไหม นี่คือการสนทนาภาษาไทย",
    "en": "Hello, how are you today?",
}

MULTILINGUAL_AUTO_PROMPT = (
    "नमस्कार, तुम्ही कसे आहात? हे मराठी संभाषण आहे. मला AI, तंत्रज्ञान आणि माहिती सांगा. "
    "नमस्ते, आप कैसे हैं? यह हिंदी वार्तालाप है। "
    "Hello, how are you today? This is an English conversation. "
    "Hola, ¿cómo estás? Bonjour, comment allez-vous? "
    "こんにちは、お元気ですか？"
)

NON_LATIN_LANGS = {
    "mr", "hi", "bn", "ta", "te", "gu", "kn", "ml", "pa", "ur",
    "ar", "ru", "ja", "zh", "ko", "th", "uk", "fa", "he", "el"
}

MARATHI_MARKERS = {
    "आहे", "आहेत", "नाही", "नाहीत", "माझं", "माझे", "माझ्या", "माझी",
    "तुझं", "तुझे", "तुझ्या", "तुझी", "तुम्हाला", "मला", "त्यांना", "आपल्याला",
    "विषयी", "बद्दल", "माहिती", "हवी", "हवे", "हवा", "कसे", "कशी", "कसा",
    "काय", "का", "सांगा", "बोलायचे", "बोलायचं", "बोलायच्या", "करायचे", "करायचं",
    "नमस्कार", "धन्यवाद", "पुन्हा", "कळलं", "नसेल", "असेल", "होते", "होती",
    "होता", "झाले", "झाली", "झाला", "करायला", "द्या", "घे", "बघू", "पाहिजे",
    "करायची", "सांगितलं", "कशा", "कशासाठी", "कधी", "कुठे", "कोण", "कोणते"
}

HINDI_MARKERS = {
    "है", "हैं", "नहीं", "था", "थी", "थे", "मेरा", "मेरी", "मेरे", "मुझे",
    "मुझको", "तेरा", "तेरी", "तेरे", "तुझे", "तुम्हारा", "तुम्हारी", "तुम्हारे",
    "आपका", "आपकी", "आपके", "आपको", "बारे", "जानकारी", "चाहिए", "कैसे", "कैसी",
    "कैसा", "क्या", "बताइए", "करना", "करिए", "नमस्ते", "होगा", "होगी", "बात", "करें"
}

LATIN_INDIC_PHONETICS = {
    "tumhala", "mazha", "majha", "tuzhya", "vishayi", "mahiti", "havi",
    "aahe", "nahi", "mala", "kase", "aahat", "sang", "bolayche", "namaskar",
    "kaisa", "kaise", "apka", "aapka", "kya", "chahiye", "batao", "dhanyawad"
}

def is_hallucinated_repetition(text: str) -> bool:
    """
    Detects Whisper repetition loops and hallucinations on background noise/silence
    such as 'ॐ ॐ ॐ ॐ ॐ...', '... ... ...', or single repeated words/syllables.
    """
    if not text or not text.strip():
        return True

    clean = text.strip()

    # 1. Repeating single character/symbol (e.g. ॐ repeated many times)
    non_space = re.sub(r"\s+", "", clean)
    if len(non_space) >= 3:
        from collections import Counter
        char_counts = Counter(non_space)
        top_char, count = char_counts.most_common(1)[0]
        if count >= 3 and (count / len(non_space)) >= 0.65:
            return True

    # 2. Repeating single word (e.g. 'ॐ ॐ ॐ' or 'thank you thank you thank you')
    words = clean.split()
    if len(words) >= 3:
        from collections import Counter
        word_counts = Counter(w.lower() for w in words)
        top_word, w_count = word_counts.most_common(1)[0]
        if w_count >= 3 and (w_count / len(words)) >= 0.60:
            return True

    # 3. Repeated short pattern (e.g. 'abc abc abc abc')
    if len(non_space) >= 8 and re.search(r"(.{2,15}?)\1{3,}", non_space):
        return True

    return False


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

        logger.info("Loading faster-whisper '%s' on %s (%s)...", self.model_name, self.device, self.compute_type)
        cpu_th = max(1, WHISPER_CPU_THREADS)
        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.cache_dir),
                cpu_threads=cpu_th,
            )
        except Exception as e:
            if self.device != "cpu":
                logger.warning("Failed to load on %s (%s), falling back to CPU...", self.device, e)
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.cache_dir),
                    cpu_threads=cpu_th,
                )
            else:
                raise e

        logger.info("STT Initialized successfully (%s).", self.device)

    def _normalize_wav_gain(self, audio_path: Union[str, Path]) -> Union[str, np.ndarray]:
        """
        Loads 16kHz mono WAV and applies dynamic digital gain normalization if input is quiet.
        Whisper accuracy drops sharply when laptop mic input is low-volume.
        """
        try:
            p = Path(audio_path)
            if p.suffix.lower() == ".wav" and p.exists() and p.stat().st_size > 44:
                import wave
                with wave.open(str(p), "rb") as wf:
                    if wf.getnchannels() == 1 and wf.getsampwidth() == 2:
                        frames = wf.readframes(wf.getnframes())
                        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        if len(data) > 0:
                            max_amp = float(np.max(np.abs(data)))
                            if 0.005 < max_amp < 0.6:
                                gain = min(8.0, 0.85 / max_amp)
                                return np.clip(data * gain, -1.0, 1.0)
                            return data
        except Exception as e:
            logger.debug("Gain normalization skipped for %s: %s", audio_path, e)
        return str(audio_path)

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        beam_size: int = 1,
    ) -> str:
        """
        Transcribes an audio file or numpy float32 audio array into text.
        Guarantees native script output (Devanagari for Marathi/Hindi, etc.)
        rather than English phonetic transliteration.
        """
        if isinstance(audio, Path):
            audio = str(audio)

        # Normalize language
        if language in (None, "", "auto", "null", "undefined"):
            req_lang = None
            init_prompt = MULTILINGUAL_AUTO_PROMPT
        else:
            req_lang = str(language).strip().lower()
            if len(req_lang) > 2 and req_lang[:2] in LANGUAGE_INITIAL_PROMPTS:
                req_lang = req_lang[:2]
            init_prompt = LANGUAGE_INITIAL_PROMPTS.get(req_lang, MULTILINGUAL_AUTO_PROMPT)

        # Audio gain normalization for quiet mic captures
        target_audio = self._normalize_wav_gain(audio) if isinstance(audio, (str, Path)) else audio

        # Beam size optimization: beam >= 3 for non-English and auto-detect prevents phonetic garble
        effective_beam = beam_size
        if effective_beam <= 1 and (req_lang is None or req_lang != "en"):
            effective_beam = 3

        # For conversational audio, vad_filter=False ensures Whisper never drops subtle vowels,
        # consonants, or nasal sounds that Silero VAD may mistakenly flag as silence.
        # temperature=0.0 eliminates 5x fallback retry loops, achieving deterministic, sub-second transcription.
        # condition_on_previous_text=False prevents hallucinations and repetitive loops.
        segments, info = self.model.transcribe(
            target_audio,
            beam_size=effective_beam,
            language=req_lang,
            initial_prompt=init_prompt,
            temperature=0.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            vad_filter=False,
        )

        detected = req_lang or (getattr(info, "language", "en") if info else "en")
        self.last_detected_language = detected
        self.last_language_probability = getattr(info, "language_probability", 1.0) if info else 1.0

        transcription = " ".join([segment.text.strip() for segment in segments]).strip()
        if is_hallucinated_repetition(transcription):
            logger.warning("STT detected hallucinated repetition loop %r; discarding.", transcription[:60])
            return ""

        # Script Contamination Guard:
        # If detected language is non-Latin (like Marathi, Hindi, etc.), but Whisper outputted
        # primarily Latin/English letters (e.g. 'Malha, Atama, Respawns'), re-run with native prompt.
        current_lang = self.last_detected_language
        if current_lang in NON_LATIN_LANGS and transcription:
            latin_chars = sum(1 for c in transcription if "a" <= c.lower() <= "z")
            total_alpha = sum(1 for c in transcription if c.isalpha())
            latin_ratio = (latin_chars / total_alpha) if total_alpha > 0 else 0.0

            if latin_ratio > 0.35 and current_lang in LANGUAGE_INITIAL_PROMPTS:
                logger.info(
                    "Detected %s speech output in Latin script (ratio %.2f). Re-transcribing with native script prompt.",
                    current_lang,
                    latin_ratio,
                )
                native_prompt = LANGUAGE_INITIAL_PROMPTS[current_lang]
                re_segs, _ = self.model.transcribe(
                    target_audio,
                    beam_size=effective_beam,
                    language=current_lang,
                    initial_prompt=native_prompt,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
                native_transcription = " ".join([s.text.strip() for s in re_segs]).strip()
                if native_transcription:
                    transcription = native_transcription

        # Indic & Multilingual Dynamic Refinement Guard:
        # In Auto-Detect mode, Whisper frequently confuses closely related languages (e.g. Marathi
        # classified as Hindi, or accented speech transliterated into Latin characters).
        # We inspect the transcribed text dynamically to mirror whatever the user actually spoke.
        if req_lang is None and transcription:
            devanagari_chars = sum(1 for c in transcription if "\u0900" <= c <= "\u097f")
            if devanagari_chars >= 2:
                devanagari_words = set(re.findall(r"[\u0900-\u097f]+", transcription.lower()))
                marathi_hits = len(devanagari_words & MARATHI_MARKERS)
                hindi_hits = len(devanagari_words & HINDI_MARKERS)

                if marathi_hits > hindi_hits or (marathi_hits >= 1 and marathi_hits >= hindi_hits and self.last_detected_language in ("hi", "en", "mr")):
                    # User spoke Marathi! If Whisper decoded with Hindi or English, re-run with Marathi
                    # language model to prevent Hindi grammatical distortion (e.g. 'कर लेना' instead of 'कळलं').
                    if self.last_detected_language != "mr":
                        logger.info(
                            "Indic refinement: Speech identified as Marathi (hits: mr=%d vs hi=%d). "
                            "Overriding language '%s' -> 'mr' and re-transcribing with authentic Marathi model.",
                            marathi_hits, hindi_hits, self.last_detected_language
                        )
                        self.last_detected_language = "mr"
                        native_prompt = LANGUAGE_INITIAL_PROMPTS.get("mr", MULTILINGUAL_AUTO_PROMPT)
                        re_segs, _ = self.model.transcribe(
                            target_audio,
                            beam_size=effective_beam,
                            language="mr",
                            initial_prompt=native_prompt,
                            temperature=0.0,
                            condition_on_previous_text=False,
                            vad_filter=False,
                        )
                        re_text = " ".join([s.text.strip() for s in re_segs]).strip()
                        if re_text:
                            transcription = re_text
                    else:
                        self.last_detected_language = "mr"
                elif hindi_hits > marathi_hits or (hindi_hits >= 1 and self.last_detected_language in ("hi", "en")):
                    self.last_detected_language = "hi"
                elif self.last_detected_language not in ("mr", "hi") and devanagari_chars >= 4:
                    self.last_detected_language = "mr" if marathi_hits >= hindi_hits else "hi"

            elif self.last_detected_language == "en":
                # Check for phonetic Latin transliteration of Indic speech
                lower_text = transcription.lower()
                latin_words = set(re.findall(r"[a-z]+", lower_text))
                latin_indic_hits = len(latin_words & LATIN_INDIC_PHONETICS)
                if latin_indic_hits >= 2:
                    is_marathi = any(w in lower_text for w in ["tumhala", "mazha", "majha", "tuzhya", "vishayi", "havi", "aahe", "mala", "aahat", "sang"])
                    target_lang = "mr" if is_marathi else "hi"
                    logger.info("Detected phonetic Latin transliteration of %s (hits=%d). Re-transcribing with native script.", target_lang, latin_indic_hits)
                    self.last_detected_language = target_lang
                    native_prompt = LANGUAGE_INITIAL_PROMPTS.get(target_lang, MULTILINGUAL_AUTO_PROMPT)
                    re_segs, _ = self.model.transcribe(
                        target_audio,
                        beam_size=effective_beam,
                        language=target_lang,
                        initial_prompt=native_prompt,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        vad_filter=False,
                    )
                    re_text = " ".join([s.text.strip() for s in re_segs]).strip()
                    if re_text:
                        transcription = re_text

        if is_hallucinated_repetition(transcription):
            return ""
        return transcription

    def transcribe_with_info(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        beam_size: int = 1,
    ) -> tuple[str, str]:
        """
        Transcribes audio and returns (transcription_text, detected_language_code).
        Auto-detects language when language is None or 'auto'.
        """
        text = self.transcribe(audio, language=language, beam_size=beam_size)
        detected_lang = getattr(self, "last_detected_language", "en") or "en"
        return text, detected_lang
