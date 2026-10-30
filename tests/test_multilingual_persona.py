import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.server import app
import app.server as server
from app.tts import (
    TTSClient,
    VOICE_CATALOG,
    LANGUAGE_DEMOGRAPHIC_VOICES,
    LANGUAGE_VOICE_MAP,
    resolve_voice_for_language,
    get_voice_language,
    normalize_language_code,
)
from app.agents import AgentManager


client = TestClient(app)

# Authenticate test client with Admin JWT token
_login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
_admin_token = _login_res.json()["access_token"]
client.headers["Authorization"] = f"Bearer {_admin_token}"


def test_get_voice_language():
    assert get_voice_language("en-US-AvaNeural") == "en"
    assert get_voice_language("hi-IN-SwaraNeural") == "hi"
    assert get_voice_language("mr-IN-AarohiNeural") == "mr"
    assert get_voice_language("es-ES-ElviraNeural") == "es"
    assert get_voice_language(None) is None


def test_resolve_voice_same_language():
    # If preferred voice matches target language, keep it
    assert resolve_voice_for_language("en", preferred_voice_id="en-US-AvaNeural") == "en-US-AvaNeural"
    assert resolve_voice_for_language("hi", preferred_voice_id="hi-IN-SwaraNeural") == "hi-IN-SwaraNeural"
    assert resolve_voice_for_language("mr", preferred_voice_id="mr-IN-ManoharNeural") == "mr-IN-ManoharNeural"


def test_resolve_voice_female_persona_adaptation():
    # When a female agent persona speaks non-English languages, it should map to female regional voices
    assert resolve_voice_for_language("hi", gender="female") == "hi-IN-SwaraNeural"
    assert resolve_voice_for_language("mr", gender="female") == "mr-IN-AarohiNeural"
    assert resolve_voice_for_language("es", gender="female") == "es-ES-ElviraNeural"
    assert resolve_voice_for_language("fr", gender="female") == "fr-FR-DeniseNeural"
    assert resolve_voice_for_language("de", gender="female") == "de-DE-KatjaNeural"
    assert resolve_voice_for_language("ja", gender="female") == "ja-JP-NanamiNeural"
    assert resolve_voice_for_language("ta", gender="female") == "ta-IN-PallaviNeural"
    assert resolve_voice_for_language("te", gender="female") == "te-IN-ShrutiNeural"


def test_resolve_voice_male_persona_adaptation():
    # When a male agent persona speaks non-English languages, it should map to male regional voices
    assert resolve_voice_for_language("hi", gender="male") == "hi-IN-MadhurNeural"
    assert resolve_voice_for_language("mr", gender="male") == "mr-IN-ManoharNeural"
    assert resolve_voice_for_language("es", gender="male") == "es-ES-AlvaroNeural"
    assert resolve_voice_for_language("fr", gender="male") == "fr-FR-HenriNeural"
    assert resolve_voice_for_language("de", gender="male") == "de-DE-KillianNeural"
    assert resolve_voice_for_language("ja", gender="male") == "ja-JP-KeitaNeural"
    assert resolve_voice_for_language("ta", gender="male") == "ta-IN-ValluvarNeural"
    assert resolve_voice_for_language("te", gender="male") == "te-IN-MohanNeural"


def test_resolve_voice_inference_from_english_preferred_voice():
    # Female English voice -> infers female persona when switching to Hindi or Marathi
    assert resolve_voice_for_language("hi", preferred_voice_id="en-US-AvaNeural") == "hi-IN-SwaraNeural"
    assert resolve_voice_for_language("mr", preferred_voice_id="en-US-AvaNeural") == "mr-IN-AarohiNeural"
    assert resolve_voice_for_language("es", preferred_voice_id="en-US-AvaNeural") == "es-ES-ElviraNeural"

    # Male English voice -> infers male persona when switching to Hindi or Marathi
    assert resolve_voice_for_language("hi", preferred_voice_id="en-US-AndrewNeural") == "hi-IN-MadhurNeural"
    assert resolve_voice_for_language("mr", preferred_voice_id="en-US-AndrewNeural") == "mr-IN-ManoharNeural"
    assert resolve_voice_for_language("es", preferred_voice_id="en-US-AndrewNeural") == "es-ES-AlvaroNeural"


def test_resolve_voice_child_and_senior_demographics():
    # Child normalizes to feminine/child tone
    assert resolve_voice_for_language("hi", gender="child") == "hi-IN-SwaraNeural"
    assert resolve_voice_for_language("mr", gender="child") == "mr-IN-AarohiNeural"

    # Aged normalizes to male/aged tone
    assert resolve_voice_for_language("hi", gender="aged") == "hi-IN-MadhurNeural"
    assert resolve_voice_for_language("mr", gender="aged") == "mr-IN-ManoharNeural"


def test_resolve_voice_backwards_compatibility():
    # No gender or preferred voice defaults cleanly to standard LANGUAGE_VOICE_MAP
    for lang, expected_voice in LANGUAGE_VOICE_MAP.items():
        assert resolve_voice_for_language(lang) == expected_voice


def test_ttsclient_get_voice_for_language_with_gender():
    assert TTSClient.get_voice_for_language("hi", gender="female") == "hi-IN-SwaraNeural"
    assert TTSClient.get_voice_for_language("hi", gender="male") == "hi-IN-MadhurNeural"
    assert TTSClient.get_voice_for_language("mr", gender="female") == "mr-IN-AarohiNeural"
    assert TTSClient.get_voice_for_language("mr", gender="male") == "mr-IN-ManoharNeural"
    assert TTSClient.get_voice_for_language("es", gender="female") == "es-ES-ElviraNeural"
    assert TTSClient.get_voice_for_language("es", gender="male") == "es-ES-AlvaroNeural"


def test_catalog_endpoint_includes_language_demographics():
    res = client.get("/api/voices/catalog")
    assert res.status_code == 200
    data = res.json()
    assert "catalog" in data
    assert "demographics" in data
    assert "language_demographics" in data

    ld = data["language_demographics"]
    assert "hi" in ld
    assert ld["hi"]["female"] == "hi-IN-SwaraNeural"
    assert ld["hi"]["male"] == "hi-IN-MadhurNeural"
    assert "mr" in ld
    assert ld["mr"]["female"] == "mr-IN-AarohiNeural"
    assert ld["mr"]["male"] == "mr-IN-ManoharNeural"


def test_preview_endpoint_with_gender_and_language():
    # Ensure TTS mock exists
    if not server.tts:
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = Path("test.wav")
        server.tts = mock_tts

    res = client.post("/api/voices/preview", json={
        "voice_model": "hi-IN-SwaraNeural",
        "gender": "female",
        "language": "hi",
        "pitch": "+5Hz",
        "rate": "+0%",
        "text": "नमस्ते, यह एक पूर्वावलोकन है।"
    })
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["voice_model"] == "hi-IN-SwaraNeural"
    assert body["gender"] == "female"
    assert body["language"] == "hi"


def test_agent_voice_persona_adaptation_in_tts_endpoint():
    mgr = AgentManager.get_instance()

    # Create female agent persona
    female_agent = mgr.create_agent({
        "name": "Priya AI",
        "role": "Customer Specialist",
        "voice_config": {
            "voice_id": "en-US-AvaNeural",
            "voice_name": "Ava (Neural)",
            "gender": "female",
            "age_group": "young_adult",
            "personality": "friendly_warm",
            "pitch": "+5Hz",
            "rate": "+0%",
            "language": "en"
        }
    })

    # Create male agent persona
    male_agent = mgr.create_agent({
        "name": "Raj AI",
        "role": "Executive Advisor",
        "voice_config": {
            "voice_id": "en-US-AndrewNeural",
            "voice_name": "Andrew (Neural)",
            "gender": "male",
            "age_group": "mature_adult",
            "personality": "professional_crisp",
            "pitch": "-5Hz",
            "rate": "+0%",
            "language": "en"
        }
    })

    old_tts = server.tts
    try:
        # Synthesize Hindi with female agent ID -> should resolve to Swara
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = Path("reply.wav")
        server.tts = mock_tts

        res1 = client.post("/api/tts", json={
            "text": "नमस्ते, मैं प्रिया हूँ।",
            "language": "hi",
            "agent_id": female_agent["id"]
        })
        assert res1.status_code == 200
        # Check call args to mock_tts.synthesize
        assert mock_tts.synthesize.called
        call_kwargs1 = mock_tts.synthesize.call_args.kwargs
        assert call_kwargs1["language"] == "hi"
        assert call_kwargs1["gender"] == "female"

        # Synthesize Marathi with male agent ID -> should resolve to Manohar
        mock_tts.reset_mock()
        res2 = client.post("/api/tts", json={
            "text": "नमस्कार, मी राज आहे.",
            "language": "mr",
            "agent_id": male_agent["id"]
        })
        assert res2.status_code == 200
        call_kwargs2 = mock_tts.synthesize.call_args.kwargs
        assert call_kwargs2["language"] == "mr"
        assert call_kwargs2["gender"] == "male"

    finally:
        server.tts = old_tts
        mgr.delete_agent(female_agent["id"])
        mgr.delete_agent(male_agent["id"])
