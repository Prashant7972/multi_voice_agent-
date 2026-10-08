import pytest
import os
import json
from fastapi.testclient import TestClient

from app.server import app, org_manager, kb_manager, contacts_manager, campaign_manager
from app.contacts import COUNTRY_CODES, format_e164

client = TestClient(app)

@pytest.fixture
def auth_headers():
    res_login = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
    assert res_login.status_code == 200
    token = res_login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_country_codes_directory():
    assert len(COUNTRY_CODES) >= 10
    iso_codes = [c["iso"] for c in COUNTRY_CODES]
    assert "US" in iso_codes
    assert "IN" in iso_codes
    assert "GB" in iso_codes
    assert "AU" in iso_codes
    assert "DE" in iso_codes

def test_format_e164():
    # US standard
    assert format_e164("+1", "555-123-4567") == "+15551234567"
    assert format_e164("1", "(555) 123-4567") == "+15551234567"
    # Leading zeros strip (e.g. UK mobile 07123456789)
    assert format_e164("+44", "07123456789") == "+447123456789"
    # India standard
    assert format_e164("+91", "9876543210") == "+919876543210"
    # Already formatted E.164
    assert format_e164("+1", "+15559998888") == "+15559998888"

def test_contacts_and_groups_crud():
    test_org_id = "test-org-crud-101"
    
    # Create group
    group = contacts_manager.create_group(
        org_id=test_org_id,
        name="Q4 Enterprise Leads",
        description="High priority outbound calling list"
    )
    assert group["id"] is not None
    assert group["name"] == "Q4 Enterprise Leads"
    
    # Create contact with country code
    contact = contacts_manager.create_contact(
        org_id=test_org_id,
        name="Sarah Connor",
        country_code="+1",
        phone="555-0199",
        group_id=group["id"],
        email="sarah@sky.net",
        notes="Key decision maker"
    )
    assert contact["id"] is not None
    assert contact["full_phone"] == "+15550199"
    assert contact["group_id"] == group["id"]
    
    # Retrieve contacts with group count
    groups = contacts_manager.list_groups(test_org_id)
    assert len(groups) >= 1
    found_grp = next(g for g in groups if g["id"] == group["id"])
    assert found_grp["contact_count"] == 1
    
    contacts = contacts_manager.list_contacts(test_org_id, group_id=group["id"])
    assert len(contacts) == 1
    assert contacts[0]["name"] == "Sarah Connor"
    
    # Delete contact
    assert contacts_manager.delete_contact(test_org_id, contact["id"]) is True
    assert len(contacts_manager.list_contacts(test_org_id, group_id=group["id"])) == 0

def test_campaign_manager_lifecycle():
    test_org_id = "test-org-camp-202"
    
    # Create test group & contacts
    group = contacts_manager.create_group(test_org_id, "VIP Alpha Group")
    c1 = contacts_manager.create_contact(test_org_id, "Alice Smith", "+1", "555-1111", group_id=group["id"])
    c2 = contacts_manager.create_contact(test_org_id, "Bob Jones", "+44", "7911-2222", group_id=group["id"])
    
    # Create campaign
    camp = campaign_manager.create_campaign(
        org_id=test_org_id,
        name="Alpha VIP Outreach",
        description="Testing concurrent multi-line calling",
        scheduled_date="2026-10-15T10:00:00",
        group_id=group["id"],
        agent_id="agent-support-pro",
        concurrency_limit=2
    )
    
    assert camp["id"] is not None
    assert camp["name"] == "Alpha VIP Outreach"
    assert camp["agent_id"] == "agent-support-pro"
    assert camp["stats"]["total_contacts"] == 2
    assert camp["status"] == "scheduled"
    assert len(camp["call_logs"]) == 2
    
    # Update campaign details
    updated = campaign_manager.update_campaign(
        org_id=test_org_id,
        campaign_id=camp["id"],
        name="Alpha VIP Outreach (Revised)",
        concurrency_limit=4
    )
    assert updated["name"] == "Alpha VIP Outreach (Revised)"
    assert updated["concurrency_limit"] == 4
    
    # Delete campaign
    assert campaign_manager.delete_campaign(test_org_id, camp["id"]) is True
    assert campaign_manager.get_campaign(test_org_id, camp["id"]) is None

def test_knowledge_base_agent_presence_testing():
    test_org_id = "test-org-kb-test-303"
    
    # Ingest a source for agent-support
    src = kb_manager.add_text_source(
        org_id=test_org_id,
        title="Payment & Refund Policy",
        category="Finance",
        content="Our refund policy allows full refunds within 30 days of purchase for all enterprise tier subscriptions.",
        agent_id="agent-support-303"
    )
    
    # 1. Test presence with the designated agent -> Should be Present & Grounded
    res_pos = kb_manager.test_source_retrieval(
        org_id=test_org_id,
        source_id=src["id"],
        agent_id="agent-support-303",
        test_query="refund policy 30 days"
    )
    assert res_pos["present_in_agent"] is True
    assert res_pos["is_accessible"] is True
    assert len(res_pos["matched_chunks"]) > 0
    assert "refund" in res_pos["matched_chunks"][0]["text"].lower()
    
    # 2. Test isolation with a different agent -> Should be Access Denied / Isolated
    res_isolated = kb_manager.test_source_retrieval(
        org_id=test_org_id,
        source_id=src["id"],
        agent_id="agent-sales-different-999",
        test_query="refund policy"
    )
    assert res_isolated["present_in_agent"] is False
    assert res_isolated["is_accessible"] is False
    assert "Confidential" in res_isolated["message"] or "Isolation" in res_isolated["message"]

def test_api_country_codes_endpoint(auth_headers):
    res = client.get("/api/contacts/country-codes", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "country_codes" in data
    assert len(data["country_codes"]) >= 10

def test_api_contacts_and_campaigns_role_authorization(auth_headers):
    # Setup organization with Owner, Admin, and User
    org = org_manager.create_organization({
        "name": "Test Telephony Org",
        "creator_email": "owner@telephony.com"
    })
    org_manager.add_member(
        org_id=org["id"],
        email="admin@telephony.com",
        name="Admin User",
        role="admin",
        adder_email="owner@telephony.com"
    )
    org_manager.add_member(
        org_id=org["id"],
        email="user@telephony.com",
        name="Regular User",
        role="user",
        adder_email="owner@telephony.com"
    )
    
    # User attempts to create contact group -> 403 Forbidden
    res_fail = client.post(
        f"/api/organizations/{org['id']}/contact-groups",
        json={
            "user_email": "user@telephony.com",
            "name": "Unauthorized Group"
        },
        headers=auth_headers
    )
    assert res_fail.status_code == 403
    
    # Admin creates contact group -> 200 OK
    res_ok = client.post(
        f"/api/organizations/{org['id']}/contact-groups",
        json={
            "user_email": "admin@telephony.com",
            "name": "Authorized Admin Group",
            "description": "Approved outbound campaign group"
        },
        headers=auth_headers
    )
    assert res_ok.status_code == 200
    grp_data = res_ok.json()
    group_id = grp_data["group"]["id"]
    
    # Admin creates contact with country code selector -> 200 OK
    res_c = client.post(
        f"/api/organizations/{org['id']}/contacts",
        json={
            "user_email": "admin@telephony.com",
            "name": "John Doe",
            "country_code": "+1",
            "phone": "415-555-2671",
            "group_id": group_id,
            "email": "johndoe@example.com"
        },
        headers=auth_headers
    )
    assert res_c.status_code == 200
    c_data = res_c.json()
    assert c_data["contact"]["full_phone"] == "+14155552671"
    
    # Admin creates Campaign -> 200 OK
    res_camp = client.post(
        f"/api/organizations/{org['id']}/campaigns",
        json={
            "user_email": "admin@telephony.com",
            "name": "Q4 Outbound Push",
            "description": "Mid-screen outreach campaign",
            "scheduled_date": "2026-11-01T09:00:00",
            "group_id": group_id,
            "agent_id": "general-voice-assistant",
            "concurrency_limit": 3
        },
        headers=auth_headers
    )
    assert res_camp.status_code == 200
    camp_id = res_camp.json()["campaign"]["id"]
    
    # Launch campaign
    res_launch = client.post(
        f"/api/organizations/{org['id']}/campaigns/{camp_id}/launch",
        json={"user_email": "admin@telephony.com"},
        headers=auth_headers
    )
    assert res_launch.status_code == 200
    assert res_launch.json()["status"] in ["in_progress", "completed"]
    
    # Ingest KB source and test endpoint
    kb_src = kb_manager.add_text_source(
        org_id=org["id"],
        title="FAQ Guide",
        category="Support",
        content="Our voice platform supports up to 10 concurrent lines per campaign.",
        agent_id="general-voice-assistant"
    )
    
    res_kb_test = client.post(
        f"/api/organizations/{org['id']}/knowledge/{kb_src['id']}/test",
        json={
            "agent_id": "general-voice-assistant",
            "query": "concurrent lines campaign"
        },
        headers=auth_headers
    )
    assert res_kb_test.status_code == 200
    kb_test_data = res_kb_test.json()
    assert kb_test_data["present_in_agent"] is True
    assert len(kb_test_data["matched_chunks"]) > 0
    assert "grounded_response_preview" in kb_test_data


def test_prashant_user_role_and_permissions(auth_headers):
    """Verify that user mailprashant2o25@gmail.com has owner permissions and can create contacts and campaigns."""
    org_id = "apex-ai-innovations"
    user_email = "mailprashant2o25@gmail.com"

    # Create contact group
    res_grp = client.post(
        f"/api/organizations/{org_id}/contact-groups",
        json={
            "user_email": user_email,
            "name": "Prashant Test Group",
            "description": "Verifying owner role access"
        },
        headers=auth_headers
    )
    assert res_grp.status_code == 200
    grp = res_grp.json()["group"]

    # Create contact
    res_c = client.post(
        f"/api/organizations/{org_id}/contacts",
        json={
            "user_email": user_email,
            "name": "Alex Test",
            "country_code": "+91",
            "phone": "9876543210",
            "group_id": grp["id"],
            "email": "alex@example.com"
        },
        headers=auth_headers
    )
    assert res_c.status_code == 200
    contact = res_c.json()["contact"]
    assert contact["full_phone"] == "+919876543210"

    # Create campaign
    res_camp = client.post(
        f"/api/organizations/{org_id}/campaigns",
        json={
            "user_email": user_email,
            "name": "Prashant Test Campaign",
            "description": "Owner campaign test",
            "scheduled_date": "2026-10-20T10:00:00",
            "group_id": grp["id"],
            "agent_id": "general-voice-assistant",
            "concurrency_limit": 2
        },
        headers=auth_headers
    )
    assert res_camp.status_code == 200
    camp = res_camp.json()["campaign"]
    assert camp["name"] == "Prashant Test Campaign"

    # Clean up test campaign, contact, group
    client.delete(f"/api/organizations/{org_id}/campaigns/{camp['id']}", headers=auth_headers)
    client.delete(f"/api/organizations/{org_id}/contacts/{contact['id']}", headers=auth_headers)
    client.delete(f"/api/organizations/{org_id}/contact-groups/{grp['id']}", headers=auth_headers)
