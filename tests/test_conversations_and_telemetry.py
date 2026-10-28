import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.server import app, org_manager, conversation_manager
from app.conversations import ConversationManager, generate_conversation_id
from app.auth import UserManager, create_access_token

client = TestClient(app)

# Authenticate test client with Admin JWT token
_login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
_admin_token = _login_res.json()["access_token"]
client.headers["Authorization"] = f"Bearer {_admin_token}"


def test_conversation_manager_storage_and_defaults():
    """Verifies that ConversationManager loads seed conversations with unique IDs and tested agents."""
    cm = ConversationManager.get_instance()
    assert len(cm.conversations) >= 4, "Should have seeded default conversations"

    # Verify seed conversation structure
    first_conv = next((c for c in cm.conversations if len(c.get("messages", [])) >= 2), cm.conversations[0])
    assert "id" in first_conv
    assert "conversation_id" in first_conv
    assert first_conv["id"] == first_conv["conversation_id"]
    assert "org_id" in first_conv
    assert "agent_id" in first_conv
    assert "agent_name" in first_conv
    assert "interaction_type" in first_conv
    assert "messages" in first_conv
    assert len(first_conv["messages"]) >= 2

    # Verify tested agents aggregation
    tested = cm.get_tested_agents("apex-ai-innovations")
    assert len(tested) >= 2
    tested_agent_ids = {a["agent_id"] for a in tested}
    assert "general-voice-assistant" in tested_agent_ids
    assert "tech-assiststant" in tested_agent_ids

    # Verify each tested agent record
    for a in tested:
        assert "channels" in a
        assert "interaction_types" in a
        assert a["conversation_count"] >= 1
        assert a["avg_latency_ms"] > 0
    print("✅ ConversationManager storage, structure, and tested agents aggregation verified!")


def test_conversation_manager_turn_recording():
    """Verifies turn recording under explicit conversation IDs."""
    cm = ConversationManager.get_instance()
    test_conv_id = f"conv-test-unit-{int(time.time()*1000)}"
    org_id = "apex-ai-innovations"

    # 1. Record User turn
    turn1 = cm.record_turn(
        role="user",
        text="What are your hours of operation?",
        conversation_id=test_conv_id,
        org_id=org_id,
        agent_id="customer-support-ai",
        agent_name="Customer Support AI",
        agent_role="Support Specialist",
        source="chat",
        interaction_type="chat",
    )
    assert turn1["conversation_id"] == test_conv_id
    assert turn1["message"]["role"] == "user"
    assert turn1["conversation"]["turns_count"] == 1

    # 2. Record Assistant turn
    turn2 = cm.record_turn(
        role="assistant",
        text="We are open 24/7 for automated voice support and 9am to 6pm for live agent escalations.",
        conversation_id=test_conv_id,
        org_id=org_id,
        agent_id="customer-support-ai",
        agent_name="Customer Support AI",
        agent_role="Support Specialist",
        source="chat",
        interaction_type="chat",
        latency={"total": 0.285},
    )
    assert turn2["conversation_id"] == test_conv_id
    assert turn2["conversation"]["turns_count"] == 2
    assert turn2["conversation"]["latency_ms"] == 285

    # 3. Retrieve conversation by ID
    conv = cm.get_conversation(org_id, test_conv_id)
    assert conv is not None
    assert conv["id"] == test_conv_id
    assert len(conv["messages"]) == 2
    assert conv["messages"][0]["text"] == "What are your hours of operation?"
    assert "24/7" in conv["messages"][1]["text"]
    print("✅ Turn recording and session retrieval by ID verified!")


def test_multi_organization_conversation_isolation():
    """Verifies that conversation sessions are strictly isolated between organizations."""
    cm = ConversationManager.get_instance()
    apex_id = "apex-ai-innovations"
    beta_id = "beta-health-org-unit"

    # Record conversation in Beta
    beta_conv_id = f"conv-beta-{int(time.time()*1000)}"
    cm.record_turn(
        role="user",
        text="Is patient records access HIPAA compliant?",
        conversation_id=beta_conv_id,
        org_id=beta_id,
        agent_id="hipaa-compliance-bot",
        agent_name="HIPAA Bot",
        interaction_type="chat",
    )

    # 1. Beta conversation cannot be accessed through Apex
    conv_in_apex = cm.get_conversation(apex_id, beta_conv_id)
    assert conv_in_apex is None, "Cross-organization conversation access must return None"

    # 2. Apex conversation list must not contain Beta conversation
    apex_list = cm.list_conversations(apex_id)
    assert not any(c["id"] == beta_conv_id for c in apex_list)

    # 3. Beta conversation list contains Beta conversation
    beta_list = cm.list_conversations(beta_id)
    assert any(c["id"] == beta_conv_id for c in beta_list)
    print("✅ Multi-tenant organization conversation isolation verified!")


def test_conversations_api_endpoints_permissions_and_scoping():
    """Verifies REST endpoints for listing and inspecting conversations by ID with role-based access control."""
    org_id = "apex-ai-innovations"
    um = UserManager.get_instance()

    # 1. Unauthenticated request -> 401 Unauthorized
    unauth_client = TestClient(app)
    res_unauth = unauth_client.get(f"/api/organizations/{org_id}/conversations")
    assert res_unauth.status_code == 401, f"Expected 401, got {res_unauth.status_code}"

    # 2. Regular user (role: user in org) -> 403 Forbidden
    member_user = um.get_user_by_email("qa@apexai.io")
    if not member_user:
        member_user = um.register_user("qa@apexai.io", "password123", "QA Member", role="Member")
    member_token = create_access_token({"sub": member_user["id"], "email": member_user["email"], "role": member_user["role"]})
    member_client = TestClient(app)
    member_client.headers["Authorization"] = f"Bearer {member_token}"
    res_member = member_client.get(f"/api/organizations/{org_id}/conversations")
    assert res_member.status_code == 403, f"Expected 403 for member, got {res_member.status_code}"

    # 3. Analyst user -> 200 OK
    analyst = um.get_user_by_email("kamon52701@hiredify.com")
    if not analyst:
        analyst = um.register_user("kamon52701@hiredify.com", "password123", "Analyst User", role="Analyst")
    org_manager.assign_role(org_id, "kamon52701@hiredify.com", "analyst", assigned_by="admin@apexai.io")
    analyst_token = create_access_token({"sub": analyst["id"], "email": analyst["email"], "role": "Analyst", "org_id": org_id})
    analyst_client = TestClient(app)
    analyst_client.headers["Authorization"] = f"Bearer {analyst_token}"
    res_analyst = analyst_client.get(f"/api/organizations/{org_id}/conversations")
    assert res_analyst.status_code == 200, f"Expected 200 for analyst, got {res_analyst.status_code}"

    # 4. Admin / Owner -> 200 OK
    res_admin = client.get(f"/api/organizations/{org_id}/conversations")
    assert res_admin.status_code == 200
    data = res_admin.json()
    assert data["success"] is True
    assert data["organization_id"] == org_id
    assert "conversations" in data
    assert "tested_agents" in data
    assert len(data["conversations"]) > 0

    # Verify each conversation in response has unique ID and metadata
    first_conv = data["conversations"][0]
    conv_id = first_conv["id"]
    assert "conversation_id" in first_conv
    assert "agent_name" in first_conv
    assert "channel" in first_conv

    # 5. Get conversation by ID: GET /api/organizations/{org_id}/conversations/{conversation_id}
    res_detail = client.get(f"/api/organizations/{org_id}/conversations/{conv_id}")
    assert res_detail.status_code == 200
    detail_data = res_detail.json()
    assert detail_data["success"] is True
    assert detail_data["conversation_id"] == conv_id
    assert detail_data["conversation"]["id"] == conv_id
    assert "messages" in detail_data["conversation"]

    # 6. Non-existent conversation ID -> 404
    res_404 = client.get(f"/api/organizations/{org_id}/conversations/conv-non-existent-99999")
    assert res_404.status_code == 404

    # 7. Filtering by interaction_type
    res_chat = client.get(f"/api/organizations/{org_id}/conversations", params={"interaction_type": "chat"})
    assert res_chat.status_code == 200
    for c in res_chat.json()["conversations"]:
        assert c["interaction_type"] == "chat"

    # 8. Filtering by search keyword
    res_search = client.get(f"/api/organizations/{org_id}/conversations", params={"search": "engineering"})
    assert res_search.status_code == 200
    assert len(res_search.json()["conversations"]) >= 1
    print("✅ Conversations REST endpoints (list, get by ID, filter, search, permissions) verified!")


def test_agent_telemetry_payload_includes_conversation_ids_and_tested_agents():
    """Verifies that GET /api/organizations/{org_id}/agent-telemetry includes conversation_ids and tested agents."""
    org_id = "apex-ai-innovations"
    res = client.get(f"/api/organizations/{org_id}/agent-telemetry")
    assert res.status_code == 200
    data = res.json()

    assert "conversations" in data
    assert "tested_agents" in data
    assert len(data["conversations"]) > 0

    # Ensure all conversations have unique string IDs
    conv_ids = [c["id"] for c in data["conversations"]]
    assert len(conv_ids) == len(set(conv_ids)), "All conversation IDs must be unique"
    for cid in conv_ids:
        assert isinstance(cid, str) and cid.startswith("conv-")

    # Ensure tested agents have tracked interaction channels
    tested = data["tested_agents"]
    assert len(tested) >= 2
    for t in tested:
        assert "agent_id" in t
        assert "agent_name" in t
        assert "channels" in t
        assert len(t["channels"]) > 0
    print("✅ Agent telemetry payload contains unique conversation IDs and tested agent personas!")


def test_knowledge_testing_assigns_and_stores_conversation_id():
    """
    Verifies that testing an agent against a knowledge source:
    1. Returns a unique conversation_id in the response.
    2. Stores the test into persistent ConversationManager.
    3. The conversation can be inspected by its ID immediately via API.
    """
    org_id = "apex-ai-innovations"

    # Fetch knowledge sources
    sources_res = client.get(f"/api/organizations/{org_id}/knowledge")
    sources = sources_res.json()["sources"]
    assert len(sources) > 0
    source_id = sources[0]["id"]

    # Run knowledge test query
    test_query = f"Explain neural noise suppression in Apex Voice {int(time.time())}"
    test_res = client.post(f"/api/organizations/{org_id}/knowledge/{source_id}/test", json={
        "query": test_query,
        "mode": "chat",
        "agent_id": "general-voice-assistant",
        "user_email": "admin@apexai.io",
    })
    assert test_res.status_code == 200
    test_data = test_res.json()

    assert "conversation_id" in test_data, "Test response must return conversation_id"
    test_conv_id = test_data["conversation_id"]
    assert test_conv_id.startswith("conv-test-")

    # Fetch that exact conversation by ID from the server
    conv_res = client.get(f"/api/organizations/{org_id}/conversations/{test_conv_id}")
    assert conv_res.status_code == 200
    conv_data = conv_res.json()["conversation"]
    assert conv_data["id"] == test_conv_id
    assert conv_data["interaction_type"] == "test"
    assert conv_data["agent_name"] == "General Voice Assistant"
    assert len(conv_data["messages"]) == 2
    assert conv_data["messages"][0]["text"] == test_query
    print(f"✅ Knowledge testing assigns, stores, and allows inspection by conversation ID ({test_conv_id})!")


def test_chat_text_endpoint_returns_and_records_conversation_id():
    """Verifies that /api/chat/text returns and maintains conversation_id."""
    org_id = "apex-ai-innovations"
    user_query = "What is the primary mission of Apex AI?"

    chat_res = client.post("/api/chat/text", json={
        "message": user_query,
        "agent_id": "tech-assiststant",
        "org_id": org_id,
        "user_email": "admin@apexai.io",
    })
    assert chat_res.status_code == 200
    chat_data = chat_res.json()

    assert "conversation_id" in chat_data
    chat_conv_id = chat_data["conversation_id"]
    assert chat_conv_id.startswith("conv-chat-")

    # Verify retrieval by ID
    conv_res = client.get(f"/api/organizations/{org_id}/conversations/{chat_conv_id}")
    assert conv_res.status_code == 200
    conv_obj = conv_res.json()["conversation"]
    assert conv_obj["id"] == chat_conv_id
    assert conv_obj["agent_id"] == "tech-assiststant"
    assert len(conv_obj["messages"]) == 2
    print(f"✅ /api/chat/text returns and persists conversation ID ({chat_conv_id})!")


if __name__ == "__main__":
    test_conversation_manager_storage_and_defaults()
    test_conversation_manager_turn_recording()
    test_multi_organization_conversation_isolation()
    test_conversations_api_endpoints_permissions_and_scoping()
    test_agent_telemetry_payload_includes_conversation_ids_and_tested_agents()
    test_knowledge_testing_assigns_and_stores_conversation_id()
    test_chat_text_endpoint_returns_and_records_conversation_id()
    print("🎉 All conversation tracking and agent telemetry tests passed successfully!")
