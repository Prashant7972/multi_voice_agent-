import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.server import app, add_conversation_message

client = TestClient(app)
_login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
_admin_token = _login_res.json()["access_token"]
client.headers["Authorization"] = f"Bearer {_admin_token}"


def test_conversation_websocket_lifecycle():
    with client.websocket_connect("/ws/conversation") as websocket:
        # 1. On connection, the client receives the init payload with current messages
        init_data = websocket.receive_json()
        assert init_data["type"] == "init"
        assert "messages" in init_data
        assert isinstance(init_data["messages"], list)

        # 2. Ping-pong keepalive test
        websocket.send_text("ping")
        pong_res = websocket.receive_text()
        assert pong_res == "pong"

        # 3. Add message broadcasts new_message event
        test_text = "Testing real-time WebSocket transcript streaming"
        new_msg = add_conversation_message(
            role="user",
            text=test_text,
            source="web",
        )

        event_data = websocket.receive_json()
        assert event_data["type"] == "new_message"
        assert event_data["message"]["text"] == test_text
        assert event_data["message"]["role"] == "user"

        # 4. Clear chat broadcasts clear event
        clear_res = client.post("/api/clear")
        assert clear_res.status_code == 200

        clear_event = websocket.receive_json()
        assert clear_event["type"] == "clear"
