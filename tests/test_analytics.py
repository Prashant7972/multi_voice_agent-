import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analytics import AnalyticsManager
from app.agents import AgentManager


def test_analytics_manager_defaults():
    am = AnalyticsManager.get_instance()
    analytics = am.get_organization_analytics("apex-ai-innovations")
    assert analytics is not None
    assert analytics["org_id"] == "apex-ai-innovations"
    assert "summary" in analytics
    assert analytics["summary"]["total_calls"] > 0
    assert analytics["summary"]["total_minutes"] > 0
    assert "channels" in analytics
    assert "trend_7d" in analytics
    assert len(analytics["trend_7d"]) == 7
    print("✅ test_analytics_manager_defaults passed")


def test_analytics_manager_record_interaction():
    am = AnalyticsManager.get_instance()
    initial = am.get_organization_analytics("apex-ai-innovations")
    initial_calls = initial["summary"]["total_calls"]

    act = am.record_interaction(
        org_id="apex-ai-innovations",
        agent_id="general-voice-assistant",
        agent_name="General Voice Assistant",
        channel="Web Voice",
        duration_sec=60.0,
        latency_ms=280,
        success=True
    )
    assert act["status"] == "Completed"
    assert act["duration_sec"] == 60

    after = am.get_organization_analytics("apex-ai-innovations")
    assert after["summary"]["total_calls"] == initial_calls + 1
    assert after["recent_activity"][0]["agent_name"] == "General Voice Assistant"
    print("✅ test_analytics_manager_record_interaction passed")


def test_analytics_scoped_to_organization():
    import time
    am = AnalyticsManager.get_instance()
    test_org_id = f"test-scoped-org-{int(time.time()*1000)}"
    analytics = am.get_organization_analytics(test_org_id)
    assert analytics["org_id"] == test_org_id
    assert analytics["summary"]["total_calls"] == 0

    # Record interaction in test org
    am.record_interaction(
        org_id=test_org_id,
        agent_id="custom-bot",
        agent_name="Custom Bot",
        channel="Telephony",
        duration_sec=120.0,
        latency_ms=310
    )
    scoped = am.get_organization_analytics(test_org_id)
    assert scoped["summary"]["total_calls"] == 1
    assert scoped["channels"]["telephony"] == 1
    assert scoped["channels"]["web_voice"] == 0
    print("✅ test_analytics_scoped_to_organization passed")


if __name__ == "__main__":
    test_analytics_manager_defaults()
    test_analytics_manager_record_interaction()
    test_analytics_scoped_to_organization()
    print("🎉 All Analytics tests passed!")
