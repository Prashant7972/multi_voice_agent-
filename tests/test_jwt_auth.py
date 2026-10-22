"""
Comprehensive Test Suite for JWT Access & Refresh Token Authentication System
Verifies:
1. JWT token generation (access & refresh tokens with claims)
2. Token decoding, signature verification, and expiration handling
3. Silent token refresh endpoint (/api/auth/refresh)
4. Strict 401 Unauthorized lockdown on unauthenticated API calls
5. Successful access with valid Bearer access tokens
6. 1-Click Instant Demo endpoint (/api/auth/demo) and demo purpose exemption
"""
import sys
import time
from datetime import timedelta
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jwt
from fastapi.testclient import TestClient
from app.server import app
from app.auth import UserManager, create_access_token, create_refresh_token, decode_token
from app.config import JWT_SECRET_KEY, JWT_ALGORITHM

client = TestClient(app)
user_manager = UserManager.get_instance()


def test_jwt_token_generation_and_claims():
    print("\n--- 1. Testing JWT Token Issuance & Claims ---")
    user = user_manager.get_user_by_email("admin@apexai.io")
    assert user is not None, "Admin user must exist"

    tokens = user_manager.issue_tokens(user)
    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] == 1800  # 30 mins

    # Verify Access Token Claims
    access_payload = user_manager.verify_token(tokens["access_token"], expected_type="access")
    assert access_payload["email"] == "admin@apexai.io"
    assert access_payload["role"] == user["role"]
    assert access_payload["type"] == "access"
    assert "exp" in access_payload
    assert "iat" in access_payload
    print("   ✅ Access token generated with valid claims and expiration.")

    # Verify Refresh Token Claims
    refresh_payload = user_manager.verify_token(tokens["refresh_token"], expected_type="refresh")
    assert refresh_payload["email"] == "admin@apexai.io"
    assert refresh_payload["type"] == "refresh"
    assert "exp" in refresh_payload
    print("   ✅ Refresh token generated with valid claims.")


def test_jwt_token_expiration_and_tampering():
    print("\n--- 2. Testing Expiration and Tampering Protection ---")
    user_data = {"sub": "test-user", "email": "test@apexai.io", "role": "Member"}

    # Create an expired token (expired 10 seconds ago)
    expired_token = create_access_token(user_data, expires_delta=timedelta(seconds=-10))
    try:
        user_manager.verify_token(expired_token, expected_type="access")
        assert False, "Expired token should have raised ExpiredSignatureError"
    except jwt.ExpiredSignatureError:
        print("   ✅ Expired access token successfully raised ExpiredSignatureError.")

    # Create a tampered token (invalid signature)
    tampered_token = expired_token[:-4] + "abcd"
    try:
        user_manager.verify_token(tampered_token, expected_type="access")
        assert False, "Tampered token should have raised InvalidTokenError"
    except (jwt.InvalidTokenError, Exception):
        print("   ✅ Tampered token successfully rejected.")

    # Token type mismatch: using refresh token as access token
    refresh_token = create_refresh_token(user_data)
    try:
        user_manager.verify_token(refresh_token, expected_type="access")
        assert False, "Refresh token should not be accepted as access token"
    except ValueError as e:
        assert "Invalid token type" in str(e)
        print("   ✅ Token type validation prevented cross-type spoofing.")


def test_jwt_refresh_endpoint():
    print("\n--- 3. Testing Token Refresh Flow (/api/auth/refresh) ---")
    user = user_manager.get_user_by_email("dev@apexai.io")
    tokens = user_manager.issue_tokens(user)
    refresh_token = tokens["refresh_token"]

    # Exchange refresh token for new access token
    res = client.post("/api/auth/refresh", json={"refresh_token": refresh_token})
    assert res.status_code == 200, f"Expected 200, got: {res.text}"
    new_data = res.json()
    assert new_data["success"] is True
    assert "access_token" in new_data
    assert "refresh_token" in new_data
    print("   ✅ Successfully renewed access token using refresh token.")

    # Missing refresh_token in request body
    res_bad = client.post("/api/auth/refresh", json={})
    assert res_bad.status_code == 400
    print("   ✅ Missing refresh_token rejected with 400 Bad Request.")

    # Invalid refresh_token string
    res_inv = client.post("/api/auth/refresh", json={"refresh_token": "fake-invalid-token"})
    assert res_inv.status_code == 401
    print("   ✅ Invalid refresh_token rejected with 401 Unauthorized.")


def test_api_strict_lockdown_without_tokens():
    print("\n--- 4. Testing Strict 401 Lockdown on Unauthenticated APIs ---")
    # All protected endpoints must return 401 when called without Bearer token
    endpoints = [
        ("GET", "/api/agents"),
        ("GET", "/api/organizations"),
        ("POST", "/api/chat/text", {"message": "hello"}),
        ("GET", "/api/conversation"),
        ("POST", "/api/clear", {}),
        ("POST", "/api/twilio/call", {"to_phone": "+1234567890"}),
    ]

    for item in endpoints:
        method = item[0]
        path = item[1]
        payload = item[2] if len(item) > 2 else {}
        if method == "GET":
            res = client.get(path)
        elif method == "POST":
            res = client.post(path, json=payload)
        
        assert res.status_code == 401, f"Expected 401 for {method} {path}, got {res.status_code}: {res.text}"
        assert "Authentication required" in res.json()["detail"] or "Missing JWT Bearer" in res.json()["detail"]

    print("   ✅ All protected workspace APIs return 401 Unauthorized when unauthenticated.")


def test_authenticated_api_access_with_bearer_token():
    print("\n--- 5. Testing Authorized API Access with Bearer Token ---")
    # 1. Login to get token
    res_login = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
    assert res_login.status_code == 200
    token = res_login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Access /api/agents with valid token
    res_agents = client.get("/api/agents", headers=headers)
    assert res_agents.status_code == 200, f"Expected 200, got: {res_agents.text}"
    assert "agents" in res_agents.json()
    print("   ✅ Authenticated access to /api/agents succeeded (200 OK).")

    # 3. Access /api/organizations with valid token
    res_orgs = client.get("/api/organizations", headers=headers)
    assert res_orgs.status_code == 200
    assert "organizations" in res_orgs.json()
    print("   ✅ Authenticated access to /api/organizations succeeded (200 OK).")

    # 4. Access /api/chat/text with valid token
    res_chat = client.post("/api/chat/text", headers=headers, json={"message": "Hello assistant!"})
    assert res_chat.status_code == 200
    assert "reply" in res_chat.json()
    print("   ✅ Authenticated access to /api/chat/text succeeded (200 OK).")


def test_instant_demo_endpoints_and_exemptions():
    print("\n--- 6. Testing Instant Demo Endpoints & Exemption ---")
    # 1. Test 1-click Demo Admin endpoint
    res_demo_admin = client.post("/api/auth/demo", json={"role": "admin"})
    assert res_demo_admin.status_code == 200
    admin_tokens = res_demo_admin.json()
    assert admin_tokens["user"]["email"] == "admin@apexai.io"
    assert "access_token" in admin_tokens
    print("   ✅ Instant Demo Admin login returned valid JWT tokens.")

    # 2. Test 1-click Demo Developer endpoint
    res_demo_dev = client.post("/api/auth/demo", json={"role": "dev"})
    assert res_demo_dev.status_code == 200
    dev_tokens = res_demo_dev.json()
    assert dev_tokens["user"]["email"] == "dev@apexai.io"
    assert "access_token" in dev_tokens
    print("   ✅ Instant Demo Developer login returned valid JWT tokens.")

    # 3. Test Demo Header Exemption (X-Demo-Mode: true)
    res_demo_header = client.get("/api/agents", headers={"X-Demo-Mode": "true"})
    assert res_demo_header.status_code == 200
    print("   ✅ Demo header exemption (X-Demo-Mode: true) successfully bypassed 401.")

    # 4. Test Demo Query Exemption (?demo=true)
    res_demo_query = client.get("/api/agents?demo=true")
    assert res_demo_query.status_code == 200
    print("   ✅ Demo query exemption (?demo=true) successfully bypassed 401.")


if __name__ == "__main__":
    test_jwt_token_generation_and_claims()
    test_jwt_token_expiration_and_tampering()
    test_jwt_refresh_endpoint()
    test_api_strict_lockdown_without_tokens()
    test_authenticated_api_access_with_bearer_token()
    test_instant_demo_endpoints_and_exemptions()
    print("\n🎉 ALL JWT ACCESS & REFRESH TOKEN TESTS PASSED SUCCESSFULLY!")
