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
