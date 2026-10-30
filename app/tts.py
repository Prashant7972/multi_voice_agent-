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
        {"id": "female", "label": "Female", "desc": "Natural feminine inflection & tone"},
        {"id": "male", "label": "Male", "desc": "Resonant masculine resonance & timbre"},
        {"id": "child", "label": "Child", "desc": "High-spirited, youthful cadence (8–14 yrs)"},
        {"id": "aged", "label": "Senior / Aged", "desc": "Distinguished, warm elder wisdom (60+ yrs)"},
    ],
    "age_groups": [
        {"id": "child", "label": "Child (8–14 yrs)"},
        {"id": "young_adult", "label": "Young Adult (18–28 yrs)"},
        {"id": "mature_adult", "label": "Mature Adult (30–55 yrs)"},
        {"id": "senior", "label": "Senior / Aged (60+ yrs)"},
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
    {
        "id": "es-ES-ElviraNeural",
        "name": "Elvira",
        "display_name": "Elvira (Spanish - Warm & Clear)",
        "language": "es",
        "language_name": "Spanish (Spain)",
        "gender": "female",
        "age_group": "young_adult",
        "age_label": "Young Adult (18-28)",
        "personality": "friendly_warm",
        "personality_label": "Friendly & Warm",
        "description": "Voz femenina cálida, expresiva y profesional, ideal para atención y soporte.",
        "pitch_default": "+0Hz",
        "rate_default": "+0%",
        "sample_phrase": "¡Hola! Soy Elvira, tu asistente de voz. ¿En qué puedo ayudarte hoy?",
    },
    {
        "id": "es-ES-AlvaroNeural",
        "name": "Alvaro",
        "display_name": "Alvaro (Spanish - Executive & Confident)",
        "language": "es",
        "language_name": "Spanish (Spain)",
        "gender": "male",
        "age_group": "mature_adult",
        "age_label": "Mature Adult (30-55)",
        "personality": "professional_crisp",
        "personality_label": "Professional & Crisp",
        "description": "Voz masculina clara, equilibrada y ejecutiva para consultoría y negocios.",
        "pitch_default": "-5Hz",
        "rate_default": "+0%",
        "sample_phrase": "Buenos días. Soy Alvaro. Repasemos los objetivos de su organización.",
    },
    {
        "id": "fr-FR-DeniseNeural",
        "name": "Denise",
        "display_name": "Denise (French - Elegant & Engaging)",
        "language": "fr",
        "language_name": "French (France)",
        "gender": "female",
        "age_group": "young_adult",
        "age_label": "Young Adult (18-28)",
        "personality": "friendly_warm",
        "personality_label": "Friendly & Warm",
        "description": "Voix féminine naturelle et élégante pour l'engagement client.",
        "pitch_default": "+0Hz",
        "rate_default": "+0%",
        "sample_phrase": "Bonjour ! Je suis Denise, votre assistante vocale. Comment puis-je vous aider ?",
    },
    {
        "id": "fr-FR-HenriNeural",
        "name": "Henri",
        "display_name": "Henri (French - Articulate & Professional)",
        "language": "fr",
        "language_name": "French (France)",
        "gender": "male",
        "age_group": "mature_adult",
        "age_label": "Mature Adult (30-55)",
        "personality": "professional_crisp",
        "personality_label": "Professional & Crisp",
        "description": "Voix masculine articulée, posée et professionnelle.",
        "pitch_default": "-5Hz",
        "rate_default": "+0%",
        "sample_phrase": "Bonjour. Je suis Henri. Prêt à optimiser vos opérations professionnelles.",
    },
    {
        "id": "de-DE-KatjaNeural",
        "name": "Katja",
        "display_name": "Katja (German - Friendly & Clear)",
        "language": "de",
        "language_name": "German (Germany)",
        "gender": "female",
        "age_group": "young_adult",
        "age_label": "Young Adult (18-28)",
        "personality": "friendly_warm",
        "personality_label": "Friendly & Warm",
        "description": "Freundliche, klare und ausdrucksstarke deutsche Stimme.",
        "pitch_default": "+0Hz",
        "rate_default": "+0%",
        "sample_phrase": "Hallo! Ich bin Katja, Ihre KI-Assistentin. Wie kann ich heute behilflich sein?",
    },
    {
        "id": "de-DE-KillianNeural",
        "name": "Killian",
        "display_name": "Killian (German - Authoritative & Calm)",
        "language": "de",
        "language_name": "German (Germany)",
        "gender": "male",
        "age_group": "mature_adult",
        "age_label": "Mature Adult (30-55)",
        "personality": "professional_crisp",
        "personality_label": "Professional & Crisp",
        "description": "Souveräne und verlässliche männliche Stimme für Unternehmen.",
        "pitch_default": "-5Hz",
        "rate_default": "+0%",
        "sample_phrase": "Guten Tag. Ich bin Killian. Lassen Sie uns die wichtigsten Ziele analysieren.",
    },
    {
        "id": "ja-JP-NanamiNeural",
        "name": "Nanami",
        "display_name": "Nanami (Japanese - Polite & Warm)",
        "language": "ja",
        "language_name": "Japanese (Japan)",
        "gender": "female",
        "age_group": "young_adult",
        "age_label": "Young Adult (18-28)",
        "personality": "friendly_warm",
        "personality_label": "Friendly & Warm",
        "description": "親切で聞き取りやすい日本語の音声アシスタントです。",
        "pitch_default": "+0Hz",
        "rate_default": "+0%",
        "sample_phrase": "こんにちは！私はナナミです。何かお手伝いできることはありますか？",
    },
    {
        "id": "ja-JP-KeitaNeural",
        "name": "Keita",
        "display_name": "Keita (Japanese - Confident & Calm)",
        "language": "ja",
        "language_name": "Japanese (Japan)",
        "gender": "male",
        "age_group": "mature_adult",
        "age_label": "Mature Adult (30-55)",
        "personality": "professional_crisp",
        "personality_label": "Professional & Crisp",
        "description": "落ち着きがあり信頼感のあるビジネス向け男性音声です。",
        "pitch_default": "-5Hz",
        "rate_default": "+0%",
        "sample_phrase": "こんにちは。ケイタです。本日の業務効率化をサポートします。",
    },
    {
        "id": "ta-IN-PallaviNeural",
        "name": "Pallavi",
        "display_name": "Pallavi (Tamil - Expressive & Polite)",
        "language": "ta",
        "language_name": "Tamil (India)",
        "gender": "female",
        "age_group": "young_adult",
        "age_label": "Young Adult (18-28)",
        "personality": "friendly_warm",
        "personality_label": "Friendly & Warm",
        "description": "தெளிவான மற்றும் மென்மையான தமிழ் குரல்.",
        "pitch_default": "+0Hz",
        "rate_default": "+0%",
        "sample_phrase": "வணக்கம்! நான் பல்லவி, உங்கள் AI குரல் உதவியாளர். உங்களுக்கு எப்படி உதவ முடியும்?",
    },
    {
        "id": "ta-IN-ValluvarNeural",
        "name": "Valluvar",
        "display_name": "Valluvar (Tamil - Dignified & Resonant)",
        "language": "ta",
        "language_name": "Tamil (India)",
        "gender": "male",
        "age_group": "mature_adult",
        "age_label": "Mature Adult (30-55)",
        "personality": "professional_crisp",
        "personality_label": "Professional & Crisp",
        "description": "அமைதியான மற்றும் அதிகாரப்பூர்வ தமிழ் ஆண் குரல்.",
        "pitch_default": "-5Hz",
        "rate_default": "+0%",
        "sample_phrase": "வணக்கம். நான் வள்ளுவர். உங்கள் பணிகளுக்கு உதவ தயாராக உள்ளேன்.",
    },
    {
        "id": "te-IN-ShrutiNeural",
        "name": "Shruti",
        "display_name": "Shruti (Telugu - Melodious & Friendly)",
        "language": "te",
        "language_name": "Telugu (India)",
        "gender": "female",
        "age_group": "young_adult",
        "age_label": "Young Adult (18-28)",
        "personality": "friendly_warm",
        "personality_label": "Friendly & Warm",
        "description": "మధురమైన మరియు స్పష్టమైన తెలుగు స్వరము.",
        "pitch_default": "+0Hz",
        "rate_default": "+0%",
        "sample_phrase": "నమస్కారం! నేను శృతి, మీ ఏఐ వాయిస్ అసిస్టెంట్. ఈరోజు మీకు ఎలా సహాయపడగలను?",
    },
    {
        "id": "te-IN-MohanNeural",
        "name": "Mohan",
        "display_name": "Mohan (Telugu - Dignified & Crisp)",
        "language": "te",
        "language_name": "Telugu (India)",
        "gender": "male",
        "age_group": "mature_adult",
        "age_label": "Mature Adult (30-55)",
        "personality": "professional_crisp",
        "personality_label": "Professional & Crisp",
        "description": "గౌరవప్రదమైన మరియు స్పష్టమైన తెలుగు పురుష స్వరం.",
        "pitch_default": "-5Hz",
        "rate_default": "+0%",
        "sample_phrase": "నమస్కారం. నేను మోహన్. మీ సహాయార్థం ఇక్కడ ఉన్నాను.",
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

LANGUAGE_DEMOGRAPHIC_VOICES: Dict[str, Dict[str, str]] = {
    "en": {
        "female": "en-US-AvaNeural",
        "male": "en-US-AndrewNeural",
        "child": "en-US-AnaNeural",
        "aged": "en-US-BrianNeural",
    },
    "hi": {
        "female": "hi-IN-SwaraNeural",
        "male": "hi-IN-MadhurNeural",
    },
    "mr": {
        "female": "mr-IN-AarohiNeural",
        "male": "mr-IN-ManoharNeural",
    },
    "es": {
        "female": "es-ES-ElviraNeural",
        "male": "es-ES-AlvaroNeural",
    },
    "fr": {
        "female": "fr-FR-DeniseNeural",
        "male": "fr-FR-HenriNeural",
    },
    "de": {
        "female": "de-DE-KatjaNeural",
        "male": "de-DE-KillianNeural",
    },
    "it": {
        "female": "it-IT-ElsaNeural",
        "male": "it-IT-DiegoNeural",
    },
    "ja": {
        "female": "ja-JP-NanamiNeural",
        "male": "ja-JP-KeitaNeural",
    },
    "zh": {
        "female": "zh-CN-XiaoxiaoNeural",
        "male": "zh-CN-YunjianNeural",
    },
    "ru": {
        "female": "ru-RU-SvetlanaNeural",
        "male": "ru-RU-DmitryNeural",
    },
    "pt": {
        "female": "pt-BR-FranciscaNeural",
        "male": "pt-BR-AntonioNeural",
    },
    "ar": {
        "female": "ar-SA-ZariyahNeural",
        "male": "ar-SA-HamedNeural",
    },
    "ta": {
        "female": "ta-IN-PallaviNeural",
        "male": "ta-IN-ValluvarNeural",
    },
    "te": {
        "female": "te-IN-ShrutiNeural",
        "male": "te-IN-MohanNeural",
    },
    "bn": {
        "female": "bn-IN-TanishaaNeural",
        "male": "bn-IN-BashkarNeural",
    },
    "gu": {
        "female": "gu-IN-DhwaniNeural",
        "male": "gu-IN-NiranjanNeural",
    },
    "kn": {
        "female": "kn-IN-SapnaNeural",
        "male": "kn-IN-GaganNeural",
    },
    "ml": {
        "female": "ml-IN-SobhanaNeural",
        "male": "ml-IN-MidhunNeural",
    },
    "pa": {
        "female": "pa-IN-RaaviNeural",
        "male": "pa-IN-OjasNeural",
    },
    "ur": {
        "female": "ur-PK-UzmaNeural",
        "male": "ur-PK-AsadNeural",
    },
    "ko": {
        "female": "ko-KR-SunHiNeural",
        "male": "ko-KR-InJoonNeural",
    },
    "nl": {
        "female": "nl-NL-FennaNeural",
        "male": "nl-NL-MaartenNeural",
    },
    "tr": {
        "female": "tr-TR-EmelNeural",
        "male": "tr-TR-AhmetNeural",
    },
    "pl": {
        "female": "pl-PL-ZofiaNeural",
        "male": "pl-PL-MarekNeural",
    },
    "sv": {
        "female": "sv-SE-SofieNeural",
        "male": "sv-SE-MattiasNeural",
    },
    "id": {
        "female": "id-ID-GadisNeural",
        "male": "id-ID-ArdiNeural",
    },
    "vi": {
        "female": "vi-VN-HoaiMyNeural",
        "male": "vi-VN-NamMinhNeural",
    },
    "th": {
        "female": "th-TH-PremwadeeNeural",
        "male": "th-TH-NiwatNeural",
    },
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


def get_voice_language(voice_id: Optional[str]) -> Optional[str]:
    """Extracts 2-letter language code from voice_id (e.g. 'en-US-AvaNeural' -> 'en', 'hi-IN-SwaraNeural' -> 'hi')."""
    if not voice_id:
        return None
    parts = voice_id.split("-")
    if parts:
        return parts[0].lower()
    return None


def resolve_voice_for_language(
    target_language: str,
    preferred_voice_id: Optional[str] = None,
    gender: Optional[str] = None,
    age_group: Optional[str] = None,
    personality: Optional[str] = None,
) -> str:
    """
    Intelligently resolves the optimal neural voice for the target language while
    faithfully preserving the agent's demographic persona (gender, age, personality).

    1. If preferred_voice_id directly matches target_language, it is preserved.
    2. If preferred_voice_id is for a different language (e.g. 'en-US-AvaNeural' during Hindi speech),
       the persona's gender (female/male/child/senior) is preserved and mapped to the target language's
       corresponding demographic voice.
    3. Preserves backwards-compatibility with LANGUAGE_VOICE_MAP defaults if no persona is specified.
    """
    lang = normalize_language_code(target_language)

    # Infer gender from preferred_voice_id if gender not explicitly passed
    resolved_gender = (gender or "").strip().lower()
    if not resolved_gender and preferred_voice_id:
        for v in VOICE_CATALOG:
            if v["id"] == preferred_voice_id:
                resolved_gender = (v.get("gender") or "").lower()
                break

    # If preferred_voice_id is compatible with the target language:
    if preferred_voice_id:
        v_lang = get_voice_language(preferred_voice_id)
        if v_lang == lang:
            return preferred_voice_id
        # Microsoft Multilingual voices can speak multiple Latin-based languages
        if "Multilingual" in preferred_voice_id and lang in ("en", "es", "fr", "de", "it", "pt"):
            return preferred_voice_id

    # If target language has demographic options for gender:
    if lang in LANGUAGE_DEMOGRAPHIC_VOICES:
        demo_map = LANGUAGE_DEMOGRAPHIC_VOICES[lang]
        if resolved_gender:
            # Direct match (female, male, child, aged)
            if resolved_gender in demo_map:
                return demo_map[resolved_gender]
            # Normalize child/aged
            if resolved_gender == "child" and "female" in demo_map:
                return demo_map["female"]
            if resolved_gender == "aged" and "male" in demo_map:
                return demo_map["male"]

        # If no gender specified, use LANGUAGE_VOICE_MAP default for backwards compatibility
        if lang in LANGUAGE_VOICE_MAP:
            return LANGUAGE_VOICE_MAP[lang]
        if "female" in demo_map:
            return demo_map["female"]
        if "male" in demo_map:
            return demo_map["male"]

    # Fallback to general voice map or default
    return LANGUAGE_VOICE_MAP.get(lang, "en-US-AvaNeural")


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
    def get_voice_for_language(cls, lang: str, gender: Optional[str] = None) -> str:
        """Returns the mapped neural voice name for the given language, optionally matching demographic gender."""
        return resolve_voice_for_language(lang, gender=gender)

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
        gender: Optional[str] = None,
    ) -> Path:
        """
        Synthesize text to a WAV audio file.
        Supports explicit voice_model and demographic persona with customized
        pitch and speech rate.
        If language is English ('en') and no voice_model is requested:
          - Utilizes local Piper neural voice for lowest offline latency.
        For non-English languages or specific voice personas:
          - Automatically resolves the matching demographic voice in the target language.
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

        # Resolve voice model according to target language and persona
        selected_voice = None
        if voice_model or lang_code != "en":
            selected_voice = resolve_voice_for_language(
                target_language=lang_code,
                preferred_voice_id=voice_model,
                gender=gender,
            )

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
