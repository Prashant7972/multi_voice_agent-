"""
Unit and integration tests for Organization-Scoped Voice Studio & Agent Voice Personas.
Verifies VoiceManager lifecycle, CRUD, organization isolation, and API endpoints.
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.server import app, voice_manager, agent_manager
import app.server as server
from app.voices import VoiceManager

# Authenticate test client with Admin JWT token
client = TestClient(app)
_login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
_admin_token = _login_res.json()["access_token"]
client.headers["Authorization"] = f"Bearer {_admin_token}"

# Mock TTS synthesis if needed
if not server.tts:
    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = Path("test.wav")
    server.tts = mock_tts


def test_voice_manager_singleton_and_seed():
    """Verifies VoiceManager singleton and seeded voices for default org."""
    vm = VoiceManager.get_instance()
    assert vm is not None
    assert len(vm.voices) >= 3

    apex_voices = vm.list_voices("apex-ai-innovations")
    assert len(apex_voices) >= 3
    names = [v["name"] for v in apex_voices]
    assert any("Maya" in n for n in names)
    assert any("Priya" in n for n in names)
    assert any("Marcus" in n for n in names)


def test_voice_manager_org_isolation():
    """Verifies voices are strictly scoped and isolated by organization."""
    vm = VoiceManager.get_instance()

    org_a = f"test-org-alpha-{int(time.time())}"
    org_b = f"test-org-beta-{int(time.time())}"

    # Org A has no voices initially
    assert len(vm.list_voices(org_a)) == 0
    assert len(vm.list_voices(org_b)) == 0

    # Create voice in Org A
    voice_a = vm.create_voice(org_a, {
        "name": "Alpha Voice Advisor",
        "gender": "female",
        "personality": "professional_crisp",
        "base_voice_id": "en-US-JennyNeural",
        "pitch": "+4Hz",
        "rate": "+0%",
        "language": "en"
    })

    # Create voice in Org B
    voice_b = vm.create_voice(org_b, {
        "name": "Beta Tech Bot",
        "gender": "male",
        "personality": "energetic_sales",
        "base_voice_id": "en-US-GuyNeural",
        "pitch": "-6Hz",
        "rate": "+10%",
        "language": "en"
    })

    # Verify isolation
    alpha_list = vm.list_voices(org_a)
    beta_list = vm.list_voices(org_b)

    assert len(alpha_list) == 1
    assert alpha_list[0]["id"] == voice_a["id"]
    assert alpha_list[0]["name"] == "Alpha Voice Advisor"

    assert len(beta_list) == 1
    assert beta_list[0]["id"] == voice_b["id"]
    assert beta_list[0]["name"] == "Beta Tech Bot"

    # Cross-org lookup returns None
    assert vm.get_voice(org_a, voice_b["id"]) is None
    assert vm.get_voice(org_b, voice_a["id"]) is None

    # Cleanup
    vm.delete_voice(org_a, voice_a["id"])
    vm.delete_voice(org_b, voice_b["id"])
    assert len(vm.list_voices(org_a)) == 0
    assert len(vm.list_voices(org_b)) == 0


def test_api_get_organization_voices():
    """Verifies GET /api/organizations/{org_id}/voices endpoint returns custom voices & catalog."""
    res = client.get("/api/organizations/apex-ai-innovations/voices")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["org_id"] == "apex-ai-innovations"
    assert "custom_voices" in data
    assert len(data["custom_voices"]) >= 3
    assert "catalog" in data
    assert len(data["catalog"]) >= 6
    assert "demographics" in data


def test_api_create_and_delete_single_voice():
    """Verifies POST and single DELETE endpoints for organization voices."""
    org_id = "apex-ai-innovations"
    unique_name = f"Test Receptionist {int(time.time() * 1000)}"

    # 1. Validation check: empty name returns 400
    res_err = client.post(f"/api/organizations/{org_id}/voices", json={"name": ""})
    assert res_err.status_code == 400

    # 2. Create custom voice
    payload = {
        "name": unique_name,
        "gender": "female",
        "age_group": "young_adult",
        "personality": "friendly_warm",
        "personality_label": "Friendly & Warm",
        "base_voice_id": "en-US-AvaNeural",
        "pitch": "+8Hz",
        "rate": "+5%",
        "language": "en",
        "language_name": "English (US)",
        "sample_phrase": "Welcome to our office! How can I assist you?",
        "notes": "Custom receptionist persona"
    }
    res_create = client.post(f"/api/organizations/{org_id}/voices", json=payload)
    assert res_create.status_code == 200
    create_data = res_create.json()
    assert create_data["success"] is True
    created_voice = create_data["voice"]
    assert created_voice["name"] == unique_name
    assert created_voice["pitch"] == "+8Hz"
    assert created_voice["rate"] == "+5%"
    voice_id = created_voice["id"]

    # 3. Verify it is listed in the organization
    res_list = client.get(f"/api/organizations/{org_id}/voices")
    assert res_list.status_code == 200
    custom_ids = [v["id"] for v in res_list.json()["custom_voices"]]
    assert voice_id in custom_ids

    # 4. Delete the custom voice
    res_del = client.delete(f"/api/organizations/{org_id}/voices/{voice_id}")
    assert res_del.status_code == 200
    del_data = res_del.json()
    assert del_data["success"] is True
    assert del_data["deleted_id"] == voice_id

    # 5. Subsequent delete returns 404
    res_del_again = client.delete(f"/api/organizations/{org_id}/voices/{voice_id}")
    assert res_del_again.status_code == 404


def test_api_create_voice_with_agent_auto_assignment():
    """Verifies that creating a voice with agent_id links the persona directly to the agent."""
    org_id = "apex-ai-innovations"

    # Create test agent
    agent = agent_manager.create_agent({
        "name": "Auto Voice Assignee",
        "role": "Consultant",
        "org_id": org_id
    })
    agent_id = agent["id"]

    try:
        # Create voice linked to agent
        voice_payload = {
            "name": "Counselor Evelyn",
            "gender": "female",
            "age_group": "mature_adult",
            "personality": "empathetic_caregiver",
            "personality_label": "Empathetic Caregiver",
            "base_voice_id": "en-US-EmmaNeural",
            "pitch": "+2Hz",
            "rate": "-5%",
            "language": "en",
            "agent_id": agent_id
        }
        res = client.post(f"/api/organizations/{org_id}/voices", json=voice_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["assigned_agent"] is not None
        assert data["assigned_agent"]["voice_config"]["voice_name"] == "Counselor Evelyn"
        assert data["assigned_agent"]["voice_config"]["pitch"] == "+2Hz"

        # Verify agent in AgentManager has updated voice_config
        updated_agent = agent_manager.get_agent(agent_id)
        assert updated_agent["voice_config"]["voice_name"] == "Counselor Evelyn"
        assert updated_agent["voice_config"]["personality"] == "empathetic_caregiver"

        # Cleanup voice
        voice_id = data["voice"]["id"]
        voice_manager.delete_voice(org_id, voice_id)
    finally:
        agent_manager.delete_agent(agent_id)


def test_api_batch_delete_voices():
    """Verifies batch deletion of custom voices."""
    org_id = f"batch-test-org-{int(time.time())}"

    # Create 3 voices in temporary org
    v1 = voice_manager.create_voice(org_id, {"name": "Batch Voice 1"})
    v2 = voice_manager.create_voice(org_id, {"name": "Batch Voice 2"})
    v3 = voice_manager.create_voice(org_id, {"name": "Batch Voice 3"})

    assert len(voice_manager.list_voices(org_id)) == 3

    # Batch delete 2 of them
    res = client.request("DELETE", f"/api/organizations/{org_id}/voices", json={"voice_ids": [v1["id"], v2["id"]]})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["deleted_count"] == 2

    remaining = voice_manager.list_voices(org_id)
    assert len(remaining) == 1
    assert remaining[0]["id"] == v3["id"]

    # Clean up remaining
    voice_manager.delete_voice(org_id, v3["id"])
    assert len(voice_manager.list_voices(org_id)) == 0


def test_agent_with_custom_voice_persona():
    """Verifies creating an agent using custom voice persona details."""
    org_id = "apex-ai-innovations"
    custom_persona = {
        "voice_id": "en-US-AvaNeural",
        "voice_name": "Maya - Empathetic Care",
        "custom_voice_id": "voice-apex-maya-caregiver",
        "gender": "female",
        "age_group": "young_adult",
        "personality": "empathetic_caregiver",
        "pitch": "+6Hz",
        "rate": "-5%",
        "language": "en"
    }

    res = client.post("/api/agents", json={
        "name": "Maya Care Companion",
        "role": "Customer Support Specialist",
        "org_id": org_id,
        "voice_config": custom_persona
    })
    assert res.status_code == 200
    agent = res.json()
    agent_id = agent["id"]

    try:
        assert agent["voice_config"]["custom_voice_id"] == "voice-apex-maya-caregiver"
        assert agent["voice_config"]["pitch"] == "+6Hz"
        assert agent["voice_config"]["rate"] == "-5%"
    finally:
        agent_manager.delete_agent(agent_id)
