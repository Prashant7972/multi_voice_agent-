import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.server import app, conversation_manager, campaign_manager
from app.call_analysis import CallAnalyzer
from app.conversations import ConversationManager

client = TestClient(app)

# Authenticate test client with Admin JWT token
_login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
_admin_token = _login_res.json()["access_token"]
client.headers["Authorization"] = f"Bearer {_admin_token}"


def test_call_analyzer_heuristic_fallback_commercial_hot_lead():
    """Verifies that CallAnalyzer heuristic fallback properly identifies high commercial intent and positive sentiment."""
    analyzer = CallAnalyzer.get_instance()
    messages = [
        {"role": "assistant", "text": "Hello, welcome to Apex AI! How can I help you today?"},
        {"role": "user", "text": "Hi! We love your platform and want to purchase an enterprise license. Can we get pricing and schedule a contract demo?"},
        {"role": "assistant", "text": "Absolutely! I can arrange an enterprise demo with our sales executive. Please let me know your timeline."},
        {"role": "user", "text": "We need to sign the contract this month and our budget is approved. Thank you, fantastic support!"}
    ]

    analysis = analyzer._heuristic_fallback(messages, agent_name="Sales Assistant")
    assert analysis["sentiment"] == "positive"
    assert analysis["sentiment_score"] > 0.5
    assert analysis["lead_score"] >= 75
    assert analysis["lead_grade"] == "HOT"
    assert any("pricing" in t.lower() or "enterprise" in t.lower() for t in analysis["key_topics"])
    assert any("commercial" in tag or "enterprise" in tag for tag in analysis["crm_tags"])
    assert len(analysis["action_items"]) > 0
    assert "Summary:" in analysis["summary"] or len(analysis["summary"]) > 10


def test_call_analyzer_heuristic_fallback_negative_complaint():
    """Verifies that CallAnalyzer heuristic fallback flags churn risk and negative sentiment."""
    analyzer = CallAnalyzer.get_instance()
    messages = [
        {"role": "assistant", "text": "Apex Support. How may I assist?"},
        {"role": "user", "text": "Your system is broken and terrible! I am extremely angry and demand a full refund immediately."},
        {"role": "assistant", "text": "I apologize for the frustration. I will check your account right away."},
        {"role": "user", "text": "This is unacceptable. If this is not resolved I will cancel my subscription."}
    ]

    analysis = analyzer._heuristic_fallback(messages, agent_name="Support Agent")
    assert analysis["sentiment"] == "negative"
    assert analysis["sentiment_score"] < 0.5
    assert analysis["customer_satisfaction"] == "Low"
    assert any("risk" in tag or "refund" in tag for tag in analysis["crm_tags"])


def test_call_analyzer_empty_transcript():
    """Verifies that empty transcripts return a safe default payload."""
    analyzer = CallAnalyzer.get_instance()
    analysis = analyzer.analyze_transcript([], agent_name="Default Agent")
    assert analysis["sentiment"] == "neutral"
    assert analysis["lead_score"] == 50
    assert analysis["lead_grade"] == "WARM"
    assert isinstance(analysis["action_items"], list)
    assert isinstance(analysis["crm_tags"], list)


def test_call_analyzer_mock_llm_json_parsing():
    """Verifies LLM JSON prompt execution and parsing."""
    analyzer = CallAnalyzer.get_instance()

    mock_llm = MagicMock()
    mock_llm.chat_messages.return_value = """
    ```json
    {
        "summary": "Customer inquired about enterprise SLA and requested an immediate quote.",
        "sentiment": "positive",
        "sentiment_score": 0.85,
        "lead_score": 92,
        "lead_grade": "HOT",
        "intent": "sales_inquiry",
        "customer_satisfaction": "High",
        "key_topics": ["enterprise SLA", "quote", "pricing"],
        "action_items": ["Send enterprise quote", "Schedule sales demo"],
        "crm_tags": ["enterprise", "high-budget", "quote-requested"]
    }
    ```
    """

    messages = [
        {"role": "user", "text": "I need an enterprise quote for 500 agents today please."}
    ]

    analysis = analyzer.analyze_transcript(messages, agent_name="Enterprise Agent", llm_client=mock_llm)
    assert analysis["sentiment"] == "positive"
    assert analysis["lead_score"] == 92
    assert analysis["lead_grade"] == "HOT"
    assert "Send enterprise quote" in analysis["action_items"]
    assert "quote-requested" in analysis["crm_tags"]


def test_conversation_manager_update_and_lead_scoring_metrics():
    """Verifies ConversationManager update_analysis and get_lead_scoring_metrics aggregation."""
    cm = ConversationManager.get_instance()
    test_conv_id = f"conv-analysis-test-{int(time.time()*1000)}"
    org_id = "apex-ai-innovations"

    # 1. Record turns with explicit kwargs
    cm.record_turn(
        role="user",
        text="I would like to purchase the enterprise suite.",
        conversation_id=test_conv_id,
        org_id=org_id,
        agent_name="Sales Bot"
    )
    cm.record_turn(
        role="assistant",
        text="Great, our team will follow up with pricing.",
        conversation_id=test_conv_id,
        org_id=org_id,
        agent_name="Sales Bot"
    )

    # 2. Update analysis
    analysis_data = {
        "summary": "Enterprise purchase discussion with follow up needed.",
        "sentiment": "positive",
        "sentiment_score": 0.8,
        "lead_score": 88,
        "lead_grade": "HOT",
        "intent": "sales_purchase",
        "customer_satisfaction": "High",
        "key_topics": ["Enterprise Architecture", "Pricing & Plans"],
        "action_items": ["Follow up with pricing proposal"],
        "crm_tags": ["enterprise", "hot-lead"]
    }
    updated = cm.update_analysis(org_id, test_conv_id, analysis_data)
    assert updated is not None

    # 3. Verify get_transcript
    transcript = cm.get_transcript(org_id, test_conv_id)
    assert transcript is not None
    assert transcript["summary"] == analysis_data["summary"]
    assert transcript["sentiment"] == "positive"
    assert transcript["lead_score"] == 88
    assert transcript["lead_grade"] == "HOT"
    assert "enterprise" in transcript["crm_tags"]

    # 4. Verify list_call_logs includes the new fields
    logs_res = cm.list_call_logs(org_id)
    logs = logs_res.get("logs", [])
    matching_log = next((l for l in logs if l["connection_id"] == test_conv_id), None)
    assert matching_log is not None
    assert matching_log["sentiment"] == "positive"
    assert matching_log["lead_score"] == 88
    assert matching_log["lead_grade"] == "HOT"

    # 5. Verify aggregated lead scoring metrics
    metrics = cm.get_lead_scoring_metrics(org_id)
    assert metrics["org_id"] == org_id
    assert "total_analyzed" in metrics
    assert metrics["total_analyzed"] >= 1
    assert "average_lead_score" in metrics
    assert "lead_grades" in metrics
    assert "sentiment_distribution" in metrics
    assert "top_crm_tags" in metrics
    assert metrics["lead_grades"]["hot"] >= 1


def test_campaign_manager_syncs_analysis():
    """Verifies that CampaignManager.update_call_status accepts analysis and updates the call log."""
    camp_mgr = campaign_manager
    org_id = "apex-ai-innovations"
    call_sid = f"CA-test-{int(time.time()*1000)}"

    camp = camp_mgr.create_campaign(
        org_id=org_id,
        name="Analysis Sync Campaign",
        description="Testing analysis sync",
        scheduled_date="2026-10-21",
        group_id="default_group",
        agent_id="general-voice-assistant",
        agent_name="Outbound Caller"
    )
    camp_id = camp["id"]

    # Assign call_sid to first log entry
    if camp.get("call_logs"):
        camp["call_logs"][0]["call_sid"] = call_sid
    else:
        camp["call_logs"].append({
            "call_id": "test-call-1",
            "call_sid": call_sid,
            "status": "calling",
            "contact_name": "Test Contact",
            "phone": "+15551234567"
        })
    camp_mgr.save()

    analysis_payload = {
        "summary": "Customer answered, agreed to renewal, score 80.",
        "sentiment": "positive",
        "lead_score": 80,
        "lead_grade": "HOT"
    }

    res = camp_mgr.update_call_status(
        campaign_id=camp_id,
        call_sid=call_sid,
        call_status="completed",
        duration_sec=45,
        analysis=analysis_payload
    )
    assert res is True

    campaign = camp_mgr.get_campaign(org_id, camp_id)
    assert campaign is not None
    matching_call = next((c for c in campaign.get("call_logs", []) if c.get("call_sid") == call_sid), None)
    assert matching_call is not None
    assert matching_call["sentiment"] == "positive"
    assert matching_call["lead_score"] == 80
    assert "renewal" in matching_call["summary"]


def test_api_trigger_reanalyze_endpoint():
    """Verifies POST /api/organizations/{org_id}/conversations/{conversation_id}/analyze."""
    cm = ConversationManager.get_instance()
    test_conv_id = f"conv-api-reanalyze-{int(time.time()*1000)}"
    org_id = "apex-ai-innovations"

    cm.record_turn(
        role="user",
        text="Can someone call me back about your pricing tiers?",
        conversation_id=test_conv_id,
        org_id=org_id,
        agent_name="Sales Rep"
    )
    cm.record_turn(
        role="assistant",
        text="Sure thing! An agent will call you back shortly.",
        conversation_id=test_conv_id,
        org_id=org_id,
        agent_name="Sales Rep"
    )

    res = client.post(f"/api/organizations/{org_id}/conversations/{test_conv_id}/analyze")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["conversation_id"] == test_conv_id
    assert "analysis" in data
    assert "sentiment" in data["analysis"]
    assert "lead_score" in data["analysis"]
    assert "summary" in data["analysis"]


def test_api_get_analysis_endpoint():
    """Verifies GET /api/organizations/{org_id}/conversations/{conversation_id}/analysis."""
    cm = ConversationManager.get_instance()
    test_conv_id = f"conv-api-get-{int(time.time()*1000)}"
    org_id = "apex-ai-innovations"

    cm.record_turn(
        role="user",
        text="Thank you, this software is wonderful!",
        conversation_id=test_conv_id,
        org_id=org_id,
        agent_name="Support"
    )

    res = client.get(f"/api/organizations/{org_id}/conversations/{test_conv_id}/analysis")
    assert res.status_code == 200
    data = res.json()
    assert data["conversation_id"] == test_conv_id
    assert "analysis" in data
    assert data["analysis"]["sentiment"] in ["positive", "neutral", "negative"]


def test_api_lead_scoring_analytics_endpoint():
    """Verifies GET /api/organizations/{org_id}/analytics/lead-scoring."""
    org_id = "apex-ai-innovations"
    res = client.get(f"/api/organizations/{org_id}/analytics/lead-scoring")
    assert res.status_code == 200
    json_data = res.json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["org_id"] == org_id
    assert "total_analyzed" in data
    assert "lead_grades" in data
    assert "sentiment_distribution" in data
    assert "top_crm_tags" in data


def test_api_analyze_404_nonexistent_conversation():
    """Verifies 404 is returned for an invalid conversation ID."""
    org_id = "apex-ai-innovations"
    res = client.post(f"/api/organizations/{org_id}/conversations/conv-nonexistent-9999/analyze")
    assert res.status_code == 404
