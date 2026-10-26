import re
import pytest
from app.stt import MARATHI_MARKERS, HINDI_MARKERS, LATIN_INDIC_PHONETICS


def test_marathi_vocabulary_marker_detection():
    # User's exact spoken sentence from screenshot 1:
    text_turn1 = "तुम्हाला माझं विषयी माहिती हवी आहे का।"
    words1 = set(re.findall(r"[\u0900-\u097f]+", text_turn1.lower()))
    
    marathi_hits1 = len(words1 & MARATHI_MARKERS)
    hindi_hits1 = len(words1 & HINDI_MARKERS)
    
    assert marathi_hits1 >= 4
    assert marathi_hits1 > hindi_hits1

    # User's exact spoken sentence from screenshot 2:
    text_turn2 = "तुम्हाला व्यवस्थित कर लेना से की मला तुझ्या विषयी कई बोलायच्या।"
    words2 = set(re.findall(r"[\u0900-\u097f]+", text_turn2.lower()))
    
    marathi_hits2 = len(words2 & MARATHI_MARKERS)
    hindi_hits2 = len(words2 & HINDI_MARKERS)
    
    assert marathi_hits2 >= 3
    assert marathi_hits2 > hindi_hits2


def test_hindi_vocabulary_marker_detection():
    hindi_sentence = "नमस्ते, क्या आप मुझे इसके बारे में कुछ जानकारी दे सकते हैं?"
    words = set(re.findall(r"[\u0900-\u097f]+", hindi_sentence.lower()))
    
    marathi_hits = len(words & MARATHI_MARKERS)
    hindi_hits = len(words & HINDI_MARKERS)
    
    assert hindi_hits >= 4
    assert hindi_hits > marathi_hits


def test_latin_indic_phonetics_detection():
    latin_marathi = "tumhala mazha vishayi mahiti havi aahe ka"
    words = set(re.findall(r"[a-z]+", latin_marathi.lower()))
    hits = len(words & LATIN_INDIC_PHONETICS)
    
    assert hits >= 3
    assert any(w in latin_marathi for w in ["tumhala", "mazha", "vishayi", "aahe"])


def test_english_non_indic_discrimination():
    english_sentence = "Hello assistant, can you tell me what time it is?"
    devanagari_chars = sum(1 for c in english_sentence if "\u0900" <= c <= "\u097f")
    assert devanagari_chars == 0
    
    words = set(re.findall(r"[a-z]+", english_sentence.lower()))
    indic_hits = len(words & LATIN_INDIC_PHONETICS)
    assert indic_hits == 0


def test_hallucinated_repetition_filter():
    from app.stt import is_hallucinated_repetition
    # Repeating Devanagari om chant hallucinated from noise
    assert is_hallucinated_repetition("ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ") is True
    # Repeating single character or punctuation
    assert is_hallucinated_repetition(".... .... .... ....") is True
    assert is_hallucinated_repetition("Thank you. Thank you. Thank you. Thank you.") is True
    # Genuine speech must not be flagged
    assert is_hallucinated_repetition("तुम्हाला माझं विषयी माहिती हवी आहे का।") is False
    assert is_hallucinated_repetition("नमस्ते, आप कैसे हैं?") is False
    assert is_hallucinated_repetition("Hello, what is the weather today?") is False


def test_language_refusal_interceptor():
    from app.server import is_language_refusal_or_deflection
    # Deflections from screenshot
    assert is_language_refusal_or_deflection("मी अनेक भाषा समजू शकतो, पण तुम्ही इंग्रजीमध्ये बोलू शकता का?", "en") is True
    assert is_language_refusal_or_deflection("मला माफ करा, मला तुमची भाषा समजत नाही. कृपया इंग्रजीमध्ये बोलू शकता का?", "mr") is True
    assert is_language_refusal_or_deflection("कृपया अंग्रेजी में बात करें", "hi") is True
    assert is_language_refusal_or_deflection("I don't understand your language. Can you speak in English?", "mr") is True
    # Genuine native responses must not be flagged
    assert is_language_refusal_or_deflection("हो, मी तुम्हाला ऐकत आहे. सांगा, मी तुमची काय मदत करू शकतो?", "mr") is False
    assert is_language_refusal_or_deflection("हाँ, मैं सुन रहा हूँ। बताइए, मैं आपकी क्या सहायता कर सकता हूँ?", "hi") is False
    assert is_language_refusal_or_deflection("Hello, how can I help you today?", "en") is False


def test_clean_multilingual_speech_strips_brackets_and_transliterations():
    from app.llm import clean_multilingual_speech

    # Exact Gemma 3:4b output from user's live test:
    gemma_sample1 = (
        "तुमचं नाव प्रभास आहे, हे ऐकून खूप आनंद झाला! (Tumcha naav Prabhas ahe, he aakun kupa aanud haala!) "
        "(Translation: I am very happy to hear your name is Prabhas!) Do you want to say anything else? 😊"
    )
    cleaned1 = clean_multilingual_speech(gemma_sample1, "mr")
    assert cleaned1 == "तुमचं नाव प्रभास आहे, हे ऐकून खूप आनंद झाला!"

    # Second Gemma 3:4b output from user's live test:
    gemma_sample2 = (
        "तुम्ही मला एक ग्लास पाणी द्या, प्लीज! (Tumhi mala ek ghaas paani dya, please!) "
        "(Translation: Please give me a glass of water!) Would you like to say anything else? 😊"
    )
    cleaned2 = clean_multilingual_speech(gemma_sample2, "mr")
    assert cleaned2 == "तुम्ही मला एक ग्लास पाणी द्या, प्लीज!"

    # Hindi output with parenthetical English notes
    hindi_sample = "नमस्ते, मैं आपकी पूरी सहायता करूँगा। (Translation: Hello, I will help you completely.)"
    cleaned_hi = clean_multilingual_speech(hindi_sample, "hi")
    assert cleaned_hi == "नमस्ते, मैं आपकी पूरी सहायता करूँगा।"

    # English output stays intact
    english_sample = "Hello there! I am ready to help you with your question."
    cleaned_en = clean_multilingual_speech(english_sample, "en")
    assert cleaned_en == "Hello there! I am ready to help you with your question."

