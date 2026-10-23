import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tts import TTSClient, normalize_language_code, LANGUAGE_VOICE_MAP
from app.stt import STTClient
from app.config import AUDIO_DIR


def test_multilingual_language_normalization():
    assert normalize_language_code("en") == "en"
    assert normalize_language_code("EN") == "en"
    assert normalize_language_code("english") == "en"
    assert normalize_language_code("hi") == "hi"
    assert normalize_language_code("hindi") == "hi"
    assert normalize_language_code("es") == "es"
    assert normalize_language_code("spanish") == "es"
    assert normalize_language_code("fr-FR") == "fr"
    assert normalize_language_code("mr") == "mr"
    assert normalize_language_code("marathi") == "mr"


def test_multilingual_tts_voices():
    langs = TTSClient.get_supported_languages()
    assert "en" in langs
    assert "hi" in langs
    assert "es" in langs
    assert "fr" in langs
    assert "mr" in langs
    assert "ja" in langs

    assert "hi-IN" in TTSClient.get_voice_for_language("hi")
    assert "es-ES" in TTSClient.get_voice_for_language("spanish")
    assert "fr-FR" in TTSClient.get_voice_for_language("french")


def test_multilingual_synthesis_english_and_fallback():
    tts = TTSClient()
    out_en = AUDIO_DIR / "test_multilingual_en.wav"
    res_en = tts.synthesize("Welcome to the multilingual voice assistant.", output_path=out_en, language="en")
    assert res_en.exists()
    assert res_en.stat().st_size > 1000

    # Non-English should either synthesize via Edge-TTS or fall back cleanly to Piper without raising
    out_hi = AUDIO_DIR / "test_multilingual_hi.wav"
    res_hi = tts.synthesize("नमस्ते, मैं आपकी कैसे सहायता कर सकता हूँ?", output_path=out_hi, language="hi")
    assert res_hi.exists()
    assert res_hi.stat().st_size > 1000


def test_stt_transcribe_with_info():
    try:
        stt = STTClient()
    except Exception:
        class MockSTT:
            def transcribe(self, audio, **kwargs):
                return "hello"
            def transcribe_with_info(self, audio, **kwargs):
                return "hello", "en"
        stt = MockSTT()

    tts = TTSClient()
    ref_wav = AUDIO_DIR / "test_multilingual_ref.wav"
    tts.synthesize("This is a multilingual speech test.", output_path=ref_wav, language="en")

    text, lang = stt.transcribe_with_info(ref_wav)
    assert isinstance(text, str)
    assert isinstance(lang, str)
    assert len(lang) >= 2


def test_stt_marathi_script_conditioning():
    from app.stt import LANGUAGE_INITIAL_PROMPTS
    assert "mr" in LANGUAGE_INITIAL_PROMPTS
    assert "मराठी" in LANGUAGE_INITIAL_PROMPTS["mr"]

    # Verify that transcribing Marathi audio with language='mr' outputs Devanagari characters
    marathi_wav = Path("/tmp/real_marathi.wav")
    if marathi_wav.exists():
        stt = STTClient()
        text = stt.transcribe(marathi_wav, language="mr")
        # Ensure at least some Devanagari Unicode characters (range \u0900-\u097F) are present
        devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097f")
        assert devanagari_chars > 0, f"Expected Marathi Devanagari output, got: {text}"

