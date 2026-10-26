import os
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from app.server import app, verify_twilio_webhook
from app.config import atomic_save_json
from app.llm import LLMClient

client = TestClient(app)


def test_healthz_and_api_health_endpoints():
    """Verifies that /healthz and /api/health return HTTP 200 with service readiness indicators."""
    for path in ["/healthz", "/api/health"]:
        res = client.get(path)
        assert res.status_code == 200, f"Expected 200 from {path}, got {res.status_code}"
        data = res.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data
        assert "storage" in data["services"]
        assert data["services"]["storage"] is True


def test_verify_twilio_webhook_fail_closed_in_production():
    """Verifies that verify_twilio_webhook fails closed in production when TWILIO_AUTH_TOKEN is missing."""
    class DummyRequest:
        def __init__(self):
            self.headers = {}
            self.url = "http://example.com/twiml/incoming"

    req = DummyRequest()
    orig_env = os.environ.get("ENVIRONMENT")
    orig_test = os.environ.get("PYTEST_CURRENT_TEST")
    orig_token = os.environ.get("TWILIO_AUTH_TOKEN")

    try:
        os.environ["ENVIRONMENT"] = "production"
        if "PYTEST_CURRENT_TEST" in os.environ:
            del os.environ["PYTEST_CURRENT_TEST"]
        os.environ["TWILIO_AUTH_TOKEN"] = ""
        os.environ["TWILIO_SIGNATURE_VALIDATION"] = "true"

        is_valid = verify_twilio_webhook(req, {})
        assert is_valid is False, "Should fail closed in production when TWILIO_AUTH_TOKEN is missing"
    finally:
        if orig_env is not None:
            os.environ["ENVIRONMENT"] = orig_env
        else:
            os.environ.pop("ENVIRONMENT", None)
        if orig_test is not None:
            os.environ["PYTEST_CURRENT_TEST"] = orig_test
        else:
            os.environ.pop("PYTEST_CURRENT_TEST", None)
        if orig_token is not None:
            os.environ["TWILIO_AUTH_TOKEN"] = orig_token
        else:
            os.environ.pop("TWILIO_AUTH_TOKEN", None)


def test_verify_twilio_webhook_candidate_urls_and_dev_fallback():
    """Verifies candidate URL reconstruction for reverse proxies and development environment fallback."""
    from twilio.request_validator import RequestValidator
    from starlette.datastructures import Headers, URL

    class MockRequest:
        def __init__(self, url, headers):
            self.url = URL(url)
            self.headers = Headers(headers)

    token = "test_auth_token_secret_12345"
    orig_env = os.environ.get("ENVIRONMENT")
    orig_test = os.environ.get("PYTEST_CURRENT_TEST")
    orig_token = os.environ.get("TWILIO_AUTH_TOKEN")

    try:
        os.environ["TWILIO_AUTH_TOKEN"] = token
        os.environ["TWILIO_SIGNATURE_VALIDATION"] = "true"
        if "PYTEST_CURRENT_TEST" in os.environ:
            del os.environ["PYTEST_CURRENT_TEST"]

        # 1. Twilio signed HTTPS public URL, Uvicorn receives HTTP with X-Forwarded-Proto
        validator = RequestValidator(token)
        pub_url = "https://abc.trycloudflare.com/twiml/incoming"
        params = {"CallSid": "CA123", "From": "+17372212163"}
        sig = validator.compute_signature(pub_url, params)

        req = MockRequest("http://abc.trycloudflare.com/twiml/incoming", {
            "x-twilio-signature": sig,
            "x-forwarded-proto": "https",
            "x-forwarded-host": "abc.trycloudflare.com"
        })

        assert verify_twilio_webhook(req, params) is True

        # 2. In non-production, signature mismatch allows fallback without dropping call
        os.environ["ENVIRONMENT"] = "development"
        bad_req = MockRequest("http://localhost:8080/twiml/incoming", {
            "x-twilio-signature": "invalid_sig_value"
        })
        assert verify_twilio_webhook(bad_req, params) is True

        # 3. In production, signature mismatch strictly fails closed
        os.environ["ENVIRONMENT"] = "production"
        assert verify_twilio_webhook(bad_req, params) is False

    finally:
        if orig_env is not None:
            os.environ["ENVIRONMENT"] = orig_env
        else:
            os.environ.pop("ENVIRONMENT", None)
        if orig_test is not None:
            os.environ["PYTEST_CURRENT_TEST"] = orig_test
        else:
            os.environ.pop("PYTEST_CURRENT_TEST", None)
        if orig_token is not None:
            os.environ["TWILIO_AUTH_TOKEN"] = orig_token
        else:
            os.environ.pop("TWILIO_AUTH_TOKEN", None)


def test_atomic_save_json_utility():
    """Verifies that atomic_save_json atomically persists data and leaves no leftover temp files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_path = Path(tmp_dir) / "test_data.json"
        data = {"key": "production_value", "nested": [1, 2, 3]}

        atomic_save_json(target_path, data)
        assert target_path.exists()

        with open(target_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

        leftover_tmps = list(Path(tmp_dir).glob("*.tmp*"))
        assert len(leftover_tmps) == 0


def test_llm_stateless_chat_execution():
    """Verifies that LLMClient.chat() supports stateless history execution without polluting global messages list."""
    llm = LLMClient()
    initial_count = len(llm.messages)

    custom_history = [{"role": "user", "content": "Prior message"}]
    try:
        res = llm.chat("Hello!", history=custom_history, stateless=True)
        assert isinstance(res, str)
    except Exception:
        pass

    assert len(llm.messages) == initial_count, "Stateless chat must not mutate self.messages"


def test_upload_file_size_rejection():
    """Verifies that payloads exceeding MAX_AUDIO_UPLOAD_BYTES return HTTP 413."""
    oversized_headers = {
        "content-length": str(30 * 1024 * 1024)
    }
    login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
    token = login_res.json()["access_token"]
    oversized_headers["Authorization"] = f"Bearer {token}"

    res = client.post(
        "/api/chat",
        headers=oversized_headers,
        files={"audio_file": ("test.wav", b"fake audio content", "audio/wav")}
    )
    assert res.status_code == 413
    assert "too large" in res.json()["detail"].lower()
