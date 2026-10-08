import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.server import app, org_manager, kb_manager

client = TestClient(app)

# Authenticate test client with Admin JWT token
_login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
_admin_token = _login_res.json()["access_token"]
client.headers["Authorization"] = f"Bearer {_admin_token}"


def test_ui_html_structure():
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # 1. Verification of removed email elements in sidebar footer
    assert 'id="sidebarUserCard"' not in html, "sidebarUserCard (email 1) should be removed from sidebar footer"
    assert 'id="sidebarNavSettingsEmail"' not in html, "sidebarNavSettingsEmail (email 2) should be removed from settings item"
    
    # 2. Verification of clean Settings item above Logged out
    assert 'id="sidebarNavSettings"' in html, "sidebarNavSettings should remain present"
    assert '<span>Logged out</span>' in html, "Logged out button should remain present"

    # 3. Verification of Knowledge Base navigation below Agent
    assert 'id="sidebarNavAgents"' in html
    assert 'id="sidebarNavKnowledge"' in html
    assert html.find('id="sidebarNavAgents"') < html.find('id="sidebarNavKnowledge"'), "Knowledge Base button should be below Agent button"
    assert 'id="btnSubnavKnowledge"' in html, "btnSubnavKnowledge should exist in subnav"

    # 4. Verification of Knowledge Base section & Add button
    assert 'id="orgSectionKnowledge"' in html, "orgSectionKnowledge section should exist"
    assert 'id="btnAddKnowledgeBase"' in html, "btnAddKnowledgeBase button should exist"
    assert "Add Knowledge Base for your Agent" in html, "Action text should match user requirement"
    assert 'id="knowledgeList"' in html, "knowledgeList container should exist"

    # 5. Verification of Add Knowledge Modal with 3 options
    assert 'id="addKnowledgeModal"' in html
    assert 'id="kbTabUpload"' in html, "Tab 1: Upload Documents should exist"
    assert 'id="kbTabText"' in html, "Tab 2: Type Knowledge should exist"
    assert 'id="kbTabVoice"' in html, "Tab 3: By Voice Recording should exist"
    assert 'id="kbFileInput"' in html
    assert 'multiple' in html, "Should support multiple document upload"

    # 6. Verification of Chatbot option in launchModal and viewChat
    assert 'confirmLaunchMode(\'chat\')' in html or 'confirmLaunchMode("chat")' in html, "launchModal must offer chat mode"
    assert 'Interactive Chatbot' in html
    assert 'id="viewChat"' in html, "viewChat view should exist"
    assert 'id="chatTextInput"' in html
    assert 'id="btnChatSend"' in html
    print("✅ All UI HTML structure and element requirements verified successfully!")

def test_knowledge_base_endpoints_and_permissions():
    org_id = "apex-ai-innovations"
    owner_email = "admin@apexai.io"
    unauth_email = "unauth.stranger@external.com"

    # 1. Fetch sources as any member
    res = client.get(f"/api/organizations/{org_id}/knowledge")
    assert res.status_code == 200
    data = res.json()
    assert "sources" in data
    assert "count" in data

    # 2. Add text knowledge as unauthenticated / non-admin user -> 403 Forbidden
    unauth_res = client.post(f"/api/organizations/{org_id}/knowledge/text", json={
        "title": "Unauthorized Policy",
        "content": "Secret information that should not be added.",
        "user_email": unauth_email,
    })
    assert unauth_res.status_code == 403

    # 3. Add text knowledge as Owner -> 200 OK
    auth_res = client.post(f"/api/organizations/{org_id}/knowledge/text", json={
        "title": "Enterprise Cloud Architecture",
        "content": "Apex AI systems deploy with isolated tenant containers and zero-latency Silero voice detection buffers.",
        "category": "Architecture",
        "user_email": owner_email,
    })
    assert auth_res.status_code == 200
    source_data = auth_res.json()["source"]
    source_id = source_data["id"]
    assert source_data["title"] == "Enterprise Cloud Architecture"

    # 4. Upload document as unauthenticated / non-admin user -> 403 Forbidden
    unauth_upload = client.post(
        f"/api/organizations/{org_id}/knowledge/upload",
        data={"user_email": unauth_email, "category": "Legal"},
        files=[("files", ("contract.txt", b"Standard SLA terms.", "text/plain"))]
    )
    assert unauth_upload.status_code == 403

    # 5. Upload document as Owner -> 200 OK
    auth_upload = client.post(
        f"/api/organizations/{org_id}/knowledge/upload",
        data={"user_email": owner_email, "category": "Operations"},
        files=[
            ("files", ("runbook1.txt", b"Step 1: Restart audio worker. Step 2: Check logs.", "text/plain")),
            ("files", ("runbook2.md", b"# Telephony Specs\nTwilio SIP trunk requires E.164 phone formats.", "text/markdown"))
        ]
    )
    assert auth_upload.status_code == 200
    upload_res = auth_upload.json()
    assert upload_res["count"] == 2
    uploaded_ids = [s["id"] for s in upload_res["sources"]]

    # 6. Delete knowledge source as unauthenticated user -> 403 Forbidden
    del_unauth = client.delete(
        f"/api/organizations/{org_id}/knowledge/{source_id}",
        params={"user_email": unauth_email}
    )
    assert del_unauth.status_code == 403

    # 7. Delete knowledge source as Owner -> 200 OK
    del_auth = client.delete(
        f"/api/organizations/{org_id}/knowledge/{source_id}",
        params={"user_email": owner_email}
    )
    assert del_auth.status_code == 200

    # Cleanup uploaded runbooks
    for uid in uploaded_ids:
        client.delete(f"/api/organizations/{org_id}/knowledge/{uid}", params={"user_email": owner_email})

    print("✅ Knowledge Base REST endpoints & permission security verified successfully!")

def test_analyst_and_regular_user_forbidden_from_ingestion():
    org_id = "apex-ai-innovations"
    # Ensure analyst exists in org
    org = org_manager.get_organization(org_id)
    analyst_email = "analyst@apexai.io"
    standard_user_email = "member@apexai.io"

    # Ingest text as analyst -> 403 Forbidden
    res = client.post(f"/api/organizations/{org_id}/knowledge/text", json={
        "title": "Analyst Note",
        "content": "Analyst should not be permitted to ingest knowledge.",
        "user_email": analyst_email,
    })
    assert res.status_code == 403, f"Expected 403 for analyst ingestion, got {res.status_code}"

    # Ingest text as standard member -> 403 Forbidden
    res = client.post(f"/api/organizations/{org_id}/knowledge/text", json={
        "title": "Member Note",
        "content": "Standard member should not be permitted to ingest knowledge.",
        "user_email": standard_user_email,
    })
    assert res.status_code == 403, f"Expected 403 for standard user ingestion, got {res.status_code}"

    # Upload document as analyst -> 403 Forbidden
    res = client.post(
        f"/api/organizations/{org_id}/knowledge/upload",
        data={"user_email": analyst_email, "category": "Analytics"},
        files=[("files", ("test.txt", b"Content", "text/plain"))]
    )
    assert res.status_code == 403, f"Expected 403 for analyst upload, got {res.status_code}"

    # Delete as analyst -> 403 Forbidden
    res = client.delete(
        f"/api/organizations/{org_id}/knowledge/kb-apex-enterprise-docs",
        params={"user_email": analyst_email}
    )
    assert res.status_code == 403, f"Expected 403 for analyst deletion, got {res.status_code}"
    print("✅ Ingestion strictly restricted to Owner and Admin (Analyst & Member rejected with 403) verified!")

def test_chat_text_endpoint_citations_filtering():
    org_id = "apex-ai-innovations"
    owner_email = "admin@apexai.io"

    # 1. Conversational greetings must NOT have citations
    greetings = ["Hello", "Hi there!", "Good morning", "How are you doing today?", "Thanks for your help!"]
    for g in greetings:
        res = client.post("/api/chat/text", json={
            "message": g,
            "org_id": org_id,
            "user_email": owner_email,
        })
        assert res.status_code == 200
        data = res.json()
        assert len(data.get("citations", [])) == 0, f"Greeting '{g}' should NOT return citations, got: {data.get('citations')}"

    # 2. General queries not in knowledge base must NOT have citations
    general_queries = ["What is the capital of France?", "Write a 2-line poem about the sea", "What is 2 + 2?"]
    for q in general_queries:
        res = client.post("/api/chat/text", json={
            "message": q,
            "org_id": org_id,
            "user_email": owner_email,
        })
        assert res.status_code == 200
        data = res.json()
        assert len(data.get("citations", [])) == 0, f"General query '{q}' should NOT return citations, got: {data.get('citations')}"

    print("✅ Citation suppression for greetings and non-KB queries verified successfully!")

def test_chat_text_endpoint_with_rag():
    org_id = "apex-ai-innovations"
    owner_email = "admin@apexai.io"

    # Add a distinctive knowledge source
    src = kb_manager.add_text_source(
        org_id=org_id,
        title="Apex Quantum Cryptography Guidelines",
        content="All data at rest must use QuantumResistantKey2026 cipher encryption.",
        category="Security",
        added_by=owner_email
    )

    try:
        # Chat inquiry asking about the quantum cryptography guidelines
        chat_res = client.post("/api/chat/text", json={
            "message": "What cipher must be used for all data at rest under Quantum Cryptography Guidelines?",
            "org_id": org_id,
            "user_email": owner_email,
            "agent_id": src["agent_id"],
        })
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        assert "reply" in chat_data
        assert "citations" in chat_data
        assert len(chat_data["citations"]) > 0
        assert any(c["source_id"] == src["id"] for c in chat_data["citations"])
        print("✅ Interactive Chatbot RAG endpoint verified with source citations!")
    finally:
        kb_manager.delete_source(org_id, src["id"])

def test_ui_demo_buttons_and_registration_modal():
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # 1. Quick demo buttons have been removed as requested
    assert "quickLogin('admin@apexai.io', 'admin123')" not in html, "Admin quick demo login button must be removed"
    assert "quickLogin('dev@apexai.io', 'dev123')" not in html, "Developer quick demo login button must be removed"

    # 2. Registration modal routing to organization dashboard
    assert 'id="userRegisteredDetailsModal"' in html
    assert 'Explore Organizations & Apply' in html
    assert "switchTab('org')" in html

    # 3. Clean Settings & Profile architecture with modular tabs
    assert 'id="tabSettingsProfile"' in html, "Settings profile tab must exist"
    assert 'id="tabSettingsOrg"' in html, "Settings org info tab must exist"
    assert 'id="tabSettingsMembers"' in html, "Settings active members tab must exist"
    assert 'id="tabSettingsInvite"' in html, "Settings member invite tab must exist"
    assert 'id="tabSettingsRequests"' in html, "Settings member requests tab must exist"
    assert 'id="tabSettingsNotifications"' in html, "Settings notifications tab must exist"
    assert 'id="settingsViewProfile"' in html, "Profile view container must exist"
    assert 'id="settingsViewEditOrg"' in html, "Edit Org view container must exist"
    assert 'id="settingsViewMembers"' in html, "Active Members view container must exist"
    assert 'id="settingsViewInvite"' in html, "Member Invite view container must exist"
    assert 'id="settingsViewRequests"' in html, "Member Requests view container must exist"
    assert 'id="settingsViewNotifications"' in html, "Notifications view container must exist"
    assert 'id="profileInputName"' in html, "Profile name input must exist"
    assert 'id="profileInputEmail"' in html, "Profile email input must exist"

    # 4. Auth Navigation Handlers
    assert 'async function navigateInitialUserDashboard()' in html, "navigateInitialUserDashboard function must be defined"
    assert 'function openActiveOrgAgents()' in html, "openActiveOrgAgents function must be defined"

    # 5. Agent-specific Knowledge Base selectors
    assert 'id="kbAgentFilterSelect"' in html, "Agent filter dropdown must exist on Knowledge Base"
    assert 'id="kbTargetAgentSelect"' in html, "Target Agent selector must exist in Add Knowledge modal"
    assert 'openAgentKnowledgeBase' in html, "openAgentKnowledgeBase action must exist"
    print("✅ Auth demo buttons removed, all modular settings tabs verified, auth navigation verified, and agent-specific KB UI elements verified!")

def test_agent_specific_server_endpoints_and_isolation():
    org_id = "apex-ai-innovations"
    owner_email = "admin@apexai.io"
    agent_x = "agent-x-secret"
    agent_y = "agent-y-public"

    # 1. Add confidential source for Agent X via REST API
    res_x = client.post(f"/api/organizations/{org_id}/knowledge/text", json={
        "title": "Agent X Protocol",
        "category": "Classified",
        "content": "Secret passcode for Agent X is PROJECT_VORTEX_909.",
        "user_email": owner_email,
        "agent_id": agent_x,
    })
    assert res_x.status_code == 200
    src_x = res_x.json()["source"]
    assert src_x["agent_id"] == agent_x

    # 2. Add source for Agent Y via REST API
    res_y = client.post(f"/api/organizations/{org_id}/knowledge/text", json={
        "title": "Agent Y Public Info",
        "category": "Public",
        "content": "Agent Y handles public weather forecasting inquiries.",
        "user_email": owner_email,
        "agent_id": agent_y,
    })
    assert res_y.status_code == 200
    src_y = res_y.json()["source"]
    assert src_y["agent_id"] == agent_y

    try:
        # 3. Verify GET /api/organizations/{org_id}/knowledge?agent_id=agent_x
        get_x = client.get(f"/api/organizations/{org_id}/knowledge", params={"agent_id": agent_x})
        assert get_x.status_code == 200
        sources_x = get_x.json()["sources"]
        assert any(s["id"] == src_x["id"] for s in sources_x)
        assert not any(s["id"] == src_y["id"] for s in sources_x)

        # 4. Chat query with agent_id = agent_x
        chat_x = client.post("/api/chat/text", json={
            "message": "What is the secret passcode for Agent X?",
            "org_id": org_id,
            "user_email": owner_email,
            "agent_id": agent_x,
        })
        assert chat_x.status_code == 200
        data_x = chat_x.json()
        assert len(data_x["citations"]) > 0
        assert any(c["source_id"] == src_x["id"] for c in data_x["citations"])

        # 5. STRICT ISOLATION: Exact same chat query with agent_id = agent_y MUST NOT access Agent X's secret
        chat_y = client.post("/api/chat/text", json={
            "message": "What is the secret passcode for Agent X?",
            "org_id": org_id,
            "user_email": owner_email,
            "agent_id": agent_y,
        })
        assert chat_y.status_code == 200
        data_y = chat_y.json()
        assert not any(c["source_id"] == src_x["id"] for c in data_y.get("citations", []))
        print("✅ REST endpoints & Chatbot agent-specific confidentiality isolation verified!")
    finally:
        client.delete(f"/api/organizations/{org_id}/knowledge/{src_x['id']}", params={"user_email": owner_email})
        client.delete(f"/api/organizations/{org_id}/knowledge/{src_y['id']}", params={"user_email": owner_email})

def test_chat_stream_endpoint():
    import json
    org_id = "apex-ai-innovations"
    owner_email = "admin@apexai.io"

    # 1. Empty message error check
    empty_res = client.post("/api/chat/stream", json={"message": "   "})
    assert empty_res.status_code == 400

    # 2. Add test source for streaming check
    src = kb_manager.add_text_source(
        org_id=org_id,
        title="Streaming Protocol Guidelines",
        content="Antigravity streaming uses Server-Sent Events with SSE data frames.",
        category="Streaming",
        added_by=owner_email
    )

    try:
        # 3. Stream request with conversational message
        stream_res = client.post("/api/chat/stream", json={
            "message": "Hello there! How are you doing today?",
            "org_id": org_id,
            "user_email": owner_email,
        })
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers.get("content-type", "")

        events = []
        for line in stream_res.text.split("\n\n"):
            line = line.strip()
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))

        event_types = [e.get("type") for e in events]
        assert "start" in event_types, "Stream should emit a start event"
        assert "done" in event_types, "Stream should emit a done event"

        done_event = next(e for e in events if e.get("type") == "done")
        assert "reply" in done_event and len(done_event["reply"]) > 0
        assert "latency" in done_event

        # 4. Stream request with RAG query
        rag_stream_res = client.post("/api/chat/stream", json={
            "message": "What protocol guidelines are used for Antigravity streaming?",
            "org_id": org_id,
            "user_email": owner_email,
            "agent_id": src["agent_id"],
        })
        assert rag_stream_res.status_code == 200
        rag_events = []
        for line in rag_stream_res.text.split("\n\n"):
            line = line.strip()
            if line.startswith("data:"):
                rag_events.append(json.loads(line[5:].strip()))

        rag_done = next(e for e in rag_events if e.get("type") == "done")
        assert len(rag_done.get("citations", [])) > 0
        assert any(c["source_id"] == src["id"] for c in rag_done["citations"])
        print("✅ SSE Token Streaming Chatbot endpoint verified successfully!")
    finally:
        kb_manager.delete_source(org_id, src["id"])

def test_voice_endpoint_brevity_capping():
    # Verify twiml respond voice prompt construction & capping
    from unittest.mock import MagicMock
    import app.server as server_module

    # Mock llm methods to verify max_tokens is passed
    mock_llm = MagicMock()
    mock_llm.chat_messages.return_value = "This is a crisp, two-sentence spoken response for voice."
    mock_llm.chat.return_value = "This is a crisp, two-sentence spoken response for voice."
    original_llm = server_module.llm
    server_module.llm = mock_llm

    try:
        twiml_res = client.post("/twiml/respond?agent_id=general-voice-assistant", data={
            "SpeechResult": "What is the capital of France?",
            "CallSid": "CA1234567890",
        })
        assert twiml_res.status_code == 200
        assert mock_llm.chat_messages.called or mock_llm.chat.called
        if mock_llm.chat_messages.called:
            call_args, call_kwargs = mock_llm.chat_messages.call_args
            assert call_kwargs.get("max_tokens") == 65
            messages = call_args[0]
            system_content = messages[0]["content"]
            assert "1 to 2 spoken sentences" in system_content
            assert "maximum 30 words" in system_content
        else:
            call_args, call_kwargs = mock_llm.chat.call_args
            assert call_kwargs.get("max_tokens") == 65
            prompt_text = call_args[0]
            assert "1 to 2 spoken sentences" in prompt_text
            assert "maximum 30 words" in prompt_text
        print("✅ Voice endpoint telephone brevity prompt and max_tokens=65 capping verified!")
    finally:
        server_module.llm = original_llm

def test_multi_turn_conversation_memory_window():
    from unittest.mock import MagicMock
    import app.server as server_module

    # Test that multi-turn history is sent to LLM
    mock_llm = MagicMock()
    mock_llm.chat_messages.return_value = "Your name is Alice."
    original_llm = server_module.llm
    server_module.llm = mock_llm

    try:
        # Clear previous memory
        client.post("/api/clear")

        # Turn 1: Client tells the agent its name
        res1 = client.post("/api/chat/text", json={
            "message": "My name is Alice and I am a backend engineer.",
            "agent_id": "general-voice-assistant",
            "org_id": "apex-ai-innovations",
            "user_email": "admin@apexai.io",
        })
        assert res1.status_code == 200

        # Verify that memory_manager recorded Turn 1
        session_key = "chat_general-voice-assistant_admin@apexai.io"
        window = server_module.memory_manager.get_window(session_key)
        assert len(window) == 2
        assert window[0]["role"] == "user"
        assert "Alice" in window[0]["content"]
        assert window[1]["role"] == "assistant"

        # Turn 2: Follow-up question asking "What is my name?"
        res2 = client.post("/api/chat/text", json={
            "message": "What is my name?",
            "agent_id": "general-voice-assistant",
            "org_id": "apex-ai-innovations",
            "user_email": "admin@apexai.io",
        })
        assert res2.status_code == 200

        # Check call_args of chat_messages: messages must contain Turn 1!
        call_args, call_kwargs = mock_llm.chat_messages.call_args
        messages_sent = call_args[0]
        user_messages = [m for m in messages_sent if m["role"] == "user"]
        assert len(user_messages) >= 2, f"Expected at least 2 user messages in context window, got {user_messages}"
        assert any("Alice" in m["content"] for m in user_messages), "Turn 1 context must be present in Turn 2 query window!"

        # Turn 3: Streaming follow-up with explicit history passed from frontend
        mock_llm.stream_prompt.return_value = iter(["You ", "are ", "Alice."])
        stream_res = client.post("/api/chat/stream", json={
            "message": "Confirm my role.",
            "history": [
                {"role": "user", "content": "My name is Alice and I am a backend engineer."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."}
            ],
            "agent_id": "general-voice-assistant",
            "org_id": "apex-ai-innovations",
            "user_email": "admin@apexai.io",
        })
        assert stream_res.status_code == 200
        call_args, call_kwargs = mock_llm.stream_prompt.call_args
        passed_history = call_kwargs.get("history")
        assert passed_history is not None
        assert len(passed_history) == 2
        assert passed_history[0]["content"] == "My name is Alice and I am a backend engineer."
        print("✅ Multi-turn conversation window preservation verified for both text and streaming chat!")
    finally:
        server_module.llm = original_llm


def test_track_section_simplification_and_telemetry_ui():
    """
    Verifies:
    1. Track section displays ONLY Spoken Interaction Volume and Channel Distribution.
    2. Redundant KPI cards (statTotalCalls, statAvgDuration, statAvgLatency, statSuccessRate) are removed.
    3. Messaging section contains the Agent Telemetry button at the bottom.
    4. Organization Agent Telemetry modal contains Latencies, Conversation History, and Test Runs tabs.
    5. Knowledge Base testing modal is upgraded to a dual-mode Chatbot (Chatting & Voice options).
    """
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # 1. Track Section: 7-Day Spoken Interaction Volume & Channel Distribution present
    assert 'id="orgSectionTrack"' in html
    assert 'id="chartBarsWrap"' in html, "7-Day Spoken Interaction Volume chart must be present"
    assert "Spoken Interaction Volume" in html
    assert "Channel Distribution" in html
    assert "channel-dist-list" in html or "channel-bar" in html

    # 2. Redundant KPI cards removed from Track section
    assert 'id="statTotalCalls"' not in html, "statTotalCalls card should be removed from Track section"
    assert 'id="statAvgDuration"' not in html, "statAvgDuration card should be removed from Track section"
    assert 'id="statAvgLatency"' not in html, "statAvgLatency card should be removed from Track section"
    assert 'id="statSuccessRate"' not in html, "statSuccessRate card should be removed from Track section"

    # 3. Dedicated Agent Telemetry Button (Standalone, rather than integrated inside Messaging)
    assert 'id="sidebarNavTelemetry"' in html, "Dedicated sidebarNavTelemetry button must exist"
    assert 'id="btnSubnavTelemetry"' in html, "Dedicated btnSubnavTelemetry button must exist in workspace subnav"
    assert 'id="btnTrackAgentTelemetry"' in html, "Dedicated btnTrackAgentTelemetry button must exist in Track header"
    assert html.find('id="sidebarNavMessaging"') < html.find('id="sidebarNavTelemetry"'), "Agent Telemetry button must be positioned below Messaging"
    assert 'id="messagingAgentTelemetryFooter"' not in html, "Telemetry button should be represented as standalone button rather than integrated inside Messaging"
    assert "Agent Telemetry" in html

    # 4. Agent Telemetry Modal with Latency, History, and Tests tabs
    assert 'id="orgAgentTelemetryModal"' in html, "orgAgentTelemetryModal must exist"
    assert 'id="tabTelemetryLatencies"' in html, "Latency metrics tab must exist"
    assert 'id="tabTelemetryHistory"' in html, "Conversation history tab must exist"
    assert 'id="tabTelemetryTests"' in html, "Agent test runs tab must exist"
    assert 'id="telemetryAgentsTbody"' in html
    assert 'id="telemetryConversationFeed"' in html
    assert 'id="telemetryTestsFeed"' in html

    # 5. Knowledge Base testing modal with Chatbot interface (Voice & Chatting modes)
    assert 'id="testKnowledgeModal"' in html, "testKnowledgeModal must exist"
    assert 'id="testKbChatFeed"' in html, "Chatbot feed container must exist"
    assert 'id="tabKbModeChat"' in html, "Chat mode switcher tab must exist"
    assert 'id="tabKbModeVoice"' in html, "Voice mode switcher tab must exist"
    assert 'id="testKbQueryInput"' in html, "Chat message input must exist"
    assert 'id="btnRunKbTest"' in html, "Chat send button must exist"
    assert 'id="btnKbVoiceMic"' in html, "Voice recording mic button must exist"
    print("✅ Track simplification, Standalone telemetry button, and KB Chatbot UI elements verified!")


def test_agent_telemetry_endpoint_security_and_payload():
    """
    Verifies:
    1. 401 Unauthorized when unauthenticated.
    2. 403 Forbidden for standard user role (not owner/admin/analyst).
    3. 200 OK for Admin/Owner/Analyst.
    4. Payload contains scoped agents with latencies, conversation history, and test runs.
    """
    org_id = "apex-ai-innovations"

    # 1. Unauthenticated request -> 401
    unauth_client = TestClient(app)
    res_unauth = unauth_client.get(f"/api/organizations/{org_id}/agent-telemetry")
    assert res_unauth.status_code == 401, f"Expected 401, got {res_unauth.status_code}"

    # 2. Regular user (qa@apexai.io has role: user in apex-ai-innovations) -> 403 Forbidden
    from app.auth import UserManager, create_access_token
    um = UserManager.get_instance()
    user = um.get_user_by_email("qa@apexai.io")
    if not user:
        user = um.register_user("qa@apexai.io", "password123", "QA Lead", role="Member")
    user_token = create_access_token({"sub": user["id"], "email": user["email"], "role": user["role"]})
    user_client = TestClient(app)
    user_client.headers["Authorization"] = f"Bearer {user_token}"
    res_user = user_client.get(f"/api/organizations/{org_id}/agent-telemetry")
    assert res_user.status_code == 403, f"Expected 403 for standard user, got {res_user.status_code}"

    # 3. Analyst user (kamon52701@hiredify.com has role: analyst in apex-ai-innovations) -> 200 OK
    analyst_token = create_access_token({"sub": "user-1790251130", "email": "kamon52701@hiredify.com", "role": "Member"})
    analyst_client = TestClient(app)
    analyst_client.headers["Authorization"] = f"Bearer {analyst_token}"
    res_analyst = analyst_client.get(f"/api/organizations/{org_id}/agent-telemetry")
    assert res_analyst.status_code == 200, f"Expected 200 for analyst, got {res_analyst.status_code}"

    # 4. Admin / Owner user -> 200 OK & verify payload structure
    res_admin = client.get(f"/api/organizations/{org_id}/agent-telemetry")
    assert res_admin.status_code == 200
    data = res_admin.json()
    assert data["success"] is True
    assert data["organization_id"] == org_id
    assert "agents" in data
    assert "conversations" in data
    assert "tests" in data

    # Check agent latency breakdown fields
    agents = data["agents"]
    assert len(agents) > 0
    first_agent = agents[0]
    assert "speech_latency_ms" in first_agent
    assert "llm_latency_ms" in first_agent
    assert "rag_latency_ms" in first_agent
    assert "avg_latency_ms" in first_agent
    assert first_agent["speech_latency_ms"] > 0
    assert first_agent["llm_latency_ms"] > 0

    # Check conversation history scoping
    conversations = data["conversations"]
    assert isinstance(conversations, list)
    assert len(conversations) > 0
    print("✅ Agent telemetry endpoint security (401/403/200) and payload structure verified!")


def test_agent_knowledge_chatbot_interactive_testing_without_auto_telemetry_logging():
    """
    Verifies that testing an agent against a knowledge source:
    1. Supports chat and voice verification queries.
    2. Returns grounded answers, citations, and chunks cleanly.
    3. Does NOT automatically log or pollute telemetry history with test trials (as requested by user).
    """
    org_id = "apex-ai-innovations"

    # Get count of tests before running queries
    telemetry_before = client.get(f"/api/organizations/{org_id}/agent-telemetry")
    count_before = len(telemetry_before.json()["tests"])

    # Fetch knowledge sources
    sources_res = client.get(f"/api/organizations/{org_id}/knowledge")
    sources = sources_res.json()["sources"]
    assert len(sources) > 0
    source_id = sources[0]["id"]

    # 1. Chat mode test
    chat_test_res = client.post(f"/api/organizations/{org_id}/knowledge/{source_id}/test", json={
        "query": "How are customer inquiries routed in Apex?",
        "mode": "chat",
        "user_email": "admin@apexai.io",
    })
    assert chat_test_res.status_code == 200
    chat_data = chat_test_res.json()
    assert chat_data["success"] is True
    assert "grounded_response_preview" in chat_data
    assert "matched_chunks" in chat_data
    # Ensure no auto-generated test_record is appended
    assert "test_record" not in chat_data

    # 2. Voice mode test
    voice_test_res = client.post(f"/api/organizations/{org_id}/knowledge/{source_id}/test", json={
        "query": "Verify voice latency parameters.",
        "mode": "voice",
        "user_email": "admin@apexai.io",
    })
    assert voice_test_res.status_code == 200
    voice_data = voice_test_res.json()
    assert voice_data["success"] is True
    assert "test_record" not in voice_data

    # 3. Verify that telemetry tests count has NOT increased (no auto-logging)
    telemetry_after = client.get(f"/api/organizations/{org_id}/agent-telemetry")
    count_after = len(telemetry_after.json()["tests"])
    assert count_after == count_before, f"Auto telemetry logging should be disabled! Expected {count_before}, got {count_after}"
    print("✅ Verified: Knowledge testing works interactively without auto telemetry logging!")


if __name__ == "__main__":
    test_ui_html_structure()
    test_ui_demo_buttons_and_registration_modal()
    test_knowledge_base_endpoints_and_permissions()
    test_analyst_and_regular_user_forbidden_from_ingestion()
    test_chat_text_endpoint_citations_filtering()
    test_chat_text_endpoint_with_rag()
    test_agent_specific_server_endpoints_and_isolation()
    test_chat_stream_endpoint()
    test_voice_endpoint_brevity_capping()
    test_multi_turn_conversation_memory_window()
    test_track_section_simplification_and_telemetry_ui()
    test_agent_telemetry_endpoint_security_and_payload()
    test_agent_knowledge_chatbot_interactive_testing_without_auto_telemetry_logging()
    print("🎉 All Knowledge Base, Chatbot Streaming, Voice Brevity, Memory Window, Track, and Telemetry tests passed with 100% success!")

