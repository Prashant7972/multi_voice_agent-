import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.server import app
from app.agents import AgentManager
from app.tts import VOICE_CATALOG, VOICE_DEMOGRAPHICS, format_pitch, format_rate

client = TestClient(app)

# Authenticate test client with Admin JWT token
_login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
_admin_token = _login_res.json()["access_token"]
client.headers["Authorization"] = f"Bearer {_admin_token}"

import app.server as server
from unittest.mock import MagicMock
if not server.tts:
    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = Path("test.wav")
    server.tts = mock_tts




def test_pitch_and_rate_formatters():
    # Pitch formatter
    assert format_pitch(None) == "+0Hz"
    assert format_pitch(0) == "+0Hz"
    assert format_pitch(15) == "+15Hz"
    assert format_pitch(-10) == "-10Hz"
    assert format_pitch("+20Hz") == "+20Hz"
    assert format_pitch("-5Hz") == "-5Hz"
    assert format_pitch("12Hz") == "+12Hz"

    # Rate formatter
    assert format_rate(None) == "+0%"
    assert format_rate(1.0) == "+0%"
    assert format_rate(1.2) == "+20%"
    assert format_rate(0.85) == "-15%"
    assert format_rate("+10%") == "+10%"
    assert format_rate("-20%") == "-20%"
    assert format_rate("1.15x") == "+15%"


def test_voice_catalog_and_demographics():
    response = client.get("/api/voices/catalog")
    assert response.status_code == 200
    data = response.json()

    assert "catalog" in data
    assert "demographics" in data
    assert len(data["catalog"]) >= 6

    # Verify demographics cover gender, age groups, and personalities
    demographics = data["demographics"]
    assert "genders" in demographics
    assert "age_groups" in demographics
    assert "personalities" in demographics

    gender_ids = [g["id"] for g in demographics["genders"]]
    assert "female" in gender_ids
    assert "male" in gender_ids
    assert "child" in gender_ids
    assert "aged" in gender_ids

    age_ids = [a["id"] for a in demographics["age_groups"]]
    assert "child" in age_ids
    assert "young_adult" in age_ids
    assert "mature_adult" in age_ids
    assert "senior" in age_ids


def test_agent_voice_config_crud():
    mgr = AgentManager.get_instance()

    # Create agent with voice config
    custom_voice = {
        "voice_id": "en-US-BrianNeural",
        "voice_name": "Brian (Senior)",
        "gender": "aged",
        "age_group": "senior",
        "personality": "calm_advisor",
        "pitch": "-10Hz",
        "rate": "-5%",
        "language": "en"
    }
    agent = mgr.create_agent({
        "name": "Elder Mentor",
        "role": "Senior Counselor",
        "voice_config": custom_voice
    })
    agent_id = agent["id"]

    try:
        # Verify via GET endpoint
        res = client.get(f"/api/agents/{agent_id}/voice")
        assert res.status_code == 200
        data = res.json()
        assert data["voice_config"]["voice_id"] == "en-US-BrianNeural"
        assert data["voice_config"]["gender"] == "aged"
        assert data["voice_config"]["age_group"] == "senior"

        # Update voice config via PUT endpoint
        updated_voice = {
            "voice_id": "en-US-AvaNeural",
            "voice_name": "Ava",
            "gender": "female",
            "age_group": "young_adult",
            "personality": "friendly_warm",
            "pitch": "+5Hz",
            "rate": "+10%"
        }
        res_put = client.put(f"/api/agents/{agent_id}/voice", json=updated_voice)
        assert res_put.status_code == 200
        put_data = res_put.json()
        assert put_data["voice_config"]["voice_id"] == "en-US-AvaNeural"
        assert put_data["voice_config"]["gender"] == "female"
        assert put_data["voice_config"]["pitch"] == "+5Hz"

        # Check persisted agent in manager
        loaded_agent = mgr.get_agent(agent_id)
        assert loaded_agent["voice_config"]["voice_id"] == "en-US-AvaNeural"
    finally:
        mgr.delete_agent(agent_id)


def test_voice_preview_endpoint():
    res = client.post("/api/voices/preview", json={
        "voice_model": "en-US-AvaNeural",
        "pitch": "+0Hz",
        "rate": "+0%",
        "text": "Audition test for voice studio."
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "audio_url" in data
    assert data["audio_url"].startswith("/api/audio/")
    assert data["voice_model"] == "en-US-AvaNeural"
