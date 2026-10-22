import pytest
import time
from fastapi.testclient import TestClient
from app.server import app, org_manager
from app.conversations import ConversationManager
from app.auth import create_access_token

client = TestClient(app)

_admin_token = None


def get_admin_token():
    global _admin_token
    if not _admin_token:
        res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
        if res.status_code == 200:
            _admin_token = res.json()["access_token"]
    return _admin_token


def get_member_token():
    from app.auth import UserManager
    um = UserManager.get_instance()
    member_user = um.get_user_by_email("qa@apexai.io")
    if not member_user:
        member_user = um.register_user("qa@apexai.io", "password123", "QA Member", role="Member")
    return create_access_token({"sub": member_user["id"], "email": member_user["email"], "role": member_user["role"]})


def test_analytics_dashboard_endpoint_kpis():
    """Verifies that GET /api/organizations/{org_id}/analytics/dashboard returns 12 KPI metrics and charts."""
    token = get_admin_token()
    headers = {"Authorization": f"Bearer {token}"}
    org_id = "apex-ai-innovations"

    res = client.get(f"/api/organizations/{org_id}/analytics/dashboard", headers=headers)
    assert res.status_code == 200
    data = res.json()["data"]

    # Verify all 12 KPI metrics from reference figure 1
    assert "total_calls_received" in data
    assert "total_calls_abandoned" in data
    assert "total_calls_answered" in data
    assert "total_live_calls_transferred" in data
    assert "service_level_pct" in data
    assert "abandoned_pct" in data
    assert "total_call_duration" in data
    assert "average_handling_time" in data
    assert "total_callbacks_taken" in data
    assert "peak_concurrency" in data
    assert "avg_concurrency" in data
    assert "calls_in_progress" in data
    assert "total_customer_requested_sms" in data
    assert "csat_score_pct" in data
    assert "nps_score_pct" in data

    # Verify chart structures
    assert "top_issue_types" in data
    assert isinstance(data["top_issue_types"], list)
    assert len(data["top_issue_types"]) > 0
    assert "type" in data["top_issue_types"][0]
    assert "count" in data["top_issue_types"][0]
    assert "aht_formatted" in data["top_issue_types"][0]

    assert "interaction_status_counts" in data
    assert isinstance(data["interaction_status_counts"], list)
    assert len(data["interaction_status_counts"]) > 0
    assert "status" in data["interaction_status_counts"][0]
    assert "count" in data["interaction_status_counts"][0]
    assert "color" in data["interaction_status_counts"][0]


def test_call_logs_pagination_and_search():
    """Verifies that GET /api/organizations/{org_id}/call-logs returns paginated columns matching figure 2."""
    token = get_admin_token()
    headers = {"Authorization": f"Bearer {token}"}
    org_id = "apex-ai-innovations"

    # 1. First page
    res = client.get(f"/api/organizations/{org_id}/call-logs?page=1&per_page=5", headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert body["total"] >= 5
    assert body["page"] == 1
    assert len(body["logs"]) == 5

    first_log = body["logs"][0]
    assert "connection_id" in first_log
    assert "call_start_time" in first_log
    assert "call_end_time" in first_log
    assert "total_aht" in first_log
    assert "disconnect_type" in first_log
    assert "final_interaction_type" in first_log
    assert "sla_pct" in first_log
    assert "ans_in_more_than_20_sec" in first_log

    # 2. Search filtering
    res_search = client.get(f"/api/organizations/{org_id}/call-logs?search=conv-apex-101", headers=headers)
    assert res_search.status_code == 200
    search_body = res_search.json()
    assert any(log["connection_id"] == "conv-apex-101" for log in search_body["logs"])


def test_call_transcript_endpoint():
    """Verifies that GET /api/organizations/{org_id}/call-logs/{connection_id}/transcript returns turn-by-turn dialogue."""
    token = get_admin_token()
    headers = {"Authorization": f"Bearer {token}"}
    org_id = "apex-ai-innovations"
    conn_id = "conv-apex-101"

    res = client.get(f"/api/organizations/{org_id}/call-logs/{conn_id}/transcript", headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["connection_id"] == conn_id
    transcript = body["transcript"]
    assert "messages" in transcript
    assert len(transcript["messages"]) >= 2
    assert transcript["messages"][0]["role"] in ["user", "assistant"]
    assert "text" in transcript["messages"][0]


def test_analytics_and_logs_rbac_isolation():
    """Verifies strict 401 unauthenticated and 403 member authorization for analytics and logs."""
    org_id = "apex-ai-innovations"

    # 1. Unauthenticated (401)
    res_unauth = client.get(f"/api/organizations/{org_id}/analytics/dashboard")
    assert res_unauth.status_code == 401

    res_logs_unauth = client.get(f"/api/organizations/{org_id}/call-logs")
    assert res_logs_unauth.status_code == 401

    # 2. Regular member (403 Forbidden)
    member_token = get_member_token()
    member_headers = {"Authorization": f"Bearer {member_token}"}
    res_member = client.get(f"/api/organizations/{org_id}/analytics/dashboard", headers=member_headers)
    assert res_member.status_code == 403

    res_logs_member = client.get(f"/api/organizations/{org_id}/call-logs", headers=member_headers)
    assert res_logs_member.status_code == 403


def test_multi_tenant_analytics_isolation():
    """Verifies that data for Org A is strictly separated from Org B."""
    cm = ConversationManager.get_instance()
    org_a = "apex-ai-innovations"
    org_b = "test-analytics-org-b"

    # Record turn in Org B
    cm.record_turn(
        role="user",
        text="Hello from tenant B",
        conversation_id="conv-tenant-b-test",
        org_id=org_b,
        agent_id="general-voice-assistant",
        interaction_type="voice",
    )

    data_b = cm.get_dashboard_analytics(org_b)
    assert data_b["organization_id"] == org_b
    assert data_b["total_calls_received"] == 1

    # Verify logs for B contain only B
    logs_b = cm.list_call_logs(org_b)
    assert logs_b["total"] == 1
    assert logs_b["logs"][0]["connection_id"] == "conv-tenant-b-test"

    # Verify logs for A do not contain B
    logs_a = cm.list_call_logs(org_a)
    assert all(log["connection_id"] != "conv-tenant-b-test" for log in logs_a["logs"])
