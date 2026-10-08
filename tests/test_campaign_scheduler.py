import pytest
import time
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from app.server import app, campaign_manager, contacts_manager, agent_manager

client = TestClient(app)

@pytest.fixture
def auth_headers():
    res_login = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
    assert res_login.status_code == 200
    token = res_login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

import asyncio

def test_campaign_scheduler_trigger():
    async def _test():
        test_org = "test-sched-org-1"
        grp = contacts_manager.create_group(test_org, name="Sched Leads", description="Auto test")
        cnt = contacts_manager.create_contact(test_org, name="Alice", country_code="+1", phone="5550100", group_id=grp["id"])

        # 1. Future campaign: should NOT be triggered
        future_time = (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
        camp_future = campaign_manager.create_campaign(
            org_id=test_org,
            name="Future Outreach",
            description="Testing future",
            scheduled_date=future_time,
            group_id=grp["id"],
            agent_id="general-voice-assistant",
            agent_name="General Voice Assistant"
        )
        assert camp_future["status"] == "scheduled"

        # 2. Past/due campaign: SHOULD be triggered
        past_time = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M")
        camp_due = campaign_manager.create_campaign(
            org_id=test_org,
            name="Due Outreach",
            description="Testing due",
            scheduled_date=past_time,
            group_id=grp["id"],
            agent_id="general-voice-assistant",
            agent_name="General Voice Assistant"
        )
        assert camp_due["status"] == "scheduled"

        # Run check
        triggered = await campaign_manager.check_and_trigger_scheduled()
        assert camp_due["id"] in triggered
        assert camp_future["id"] not in triggered

        # Verify status changed
        updated_due = campaign_manager.get_campaign(test_org, camp_due["id"])
        assert updated_due["status"] in ["in_progress", "completed"]

        updated_future = campaign_manager.get_campaign(test_org, camp_future["id"])
        assert updated_future["status"] == "scheduled"

    asyncio.run(_test())

def test_twilio_status_callback():
    test_org = "test-status-org-2"
    grp = contacts_manager.create_group(test_org, name="Status Leads", description="Auto test")
    cnt = contacts_manager.create_contact(test_org, name="Bob", country_code="+1", phone="5550200", group_id=grp["id"])

    camp = campaign_manager.create_campaign(
        org_id=test_org,
        name="Callback Test Campaign",
        description="Testing callback",
        scheduled_date="2026-10-01T12:00",
        group_id=grp["id"],
        agent_id="general-voice-assistant",
    )
    
    # Assign a fake CallSid
    fake_sid = "CAtest1234567890"
    camp["call_logs"][0]["call_sid"] = fake_sid
    campaign_manager.save()

    # Post to status-callback endpoint
    res = client.post(
        f"/twiml/status-callback?campaign_id={camp['id']}",
        data={
            "CallSid": fake_sid,
            "CallStatus": "completed",
            "CallDuration": "45"
        }
    )
    assert res.status_code == 200

    # Verify log entry updated
    c = campaign_manager.get_campaign(test_org, camp["id"])
    log = c["call_logs"][0]
    assert log["status"] == "completed"
    assert log["duration_sec"] == 45
    assert "45s" in log["summary"]

def test_twiml_designated_agent_endpoints():
    # 1. Incoming with designated agent
    res_in = client.post("/twiml/incoming?agent_id=general-voice-assistant")
    assert res_in.status_code == 200
    assert "xml" in res_in.headers.get("content-type", "")
    assert "General Voice Assistant" in res_in.text or "local AI voice assistant" in res_in.text
    assert "/twiml/respond?agent_id=general-voice-assistant" in res_in.text

    # 2. Respond with designated agent
    res_resp = client.post(
        "/twiml/respond?agent_id=general-voice-assistant",
        data={
            "SpeechResult": "Hello, what is your name?",
            "CallSid": "CAtestcall9999"
        }
    )
    assert res_resp.status_code == 200
    assert "xml" in res_resp.headers.get("content-type", "")
    assert "<Gather" in res_resp.text
