"""
Integration test for Authentication API endpoints in server.py
"""
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.server import app

client = TestClient(app)

def test_auth_endpoints():
    print("🧪 Testing Authentication REST Endpoints...")

    # 1. Test Login with valid Admin credentials
    res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["success"] is True
    assert data["user"]["email"] == "admin@apexai.io"
    assert "token" in data
    print("   ✅ Valid login returned 200 and session token.")

    # 2. Test Login with Developer credentials
    res_dev = client.post("/api/auth/login", json={"email": "dev@apexai.io", "password": "dev123"})
    assert res_dev.status_code == 200
    assert res_dev.json()["user"]["email"] == "dev@apexai.io"
    print("   ✅ Developer login verified.")

    # 3. Test Login with Invalid password
    res_invalid = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "wrongpassword"})
    assert res_invalid.status_code == 401
    assert "detail" in res_invalid.json()
    print("   ✅ Invalid password rejected with 401 Unauthorized.")

    # 4. Test Session check
    res_sess = client.get("/api/auth/session?email=admin@apexai.io")
    assert res_sess.status_code == 200
    assert res_sess.json()["authenticated"] is True
    assert res_sess.json()["user"]["email"] == "admin@apexai.io"
    print("   ✅ Session verification endpoint verified.")

    # 5. Test Logout
    res_logout = client.post("/api/auth/logout")
    assert res_logout.status_code == 200
    assert res_logout.json()["success"] is True
    print("   ✅ Logout endpoint verified.")

    # 6. Test Direct Registration (Used by Signup form)
    import time
    test_direct_email = f"direct-user-{int(time.time())}@voicing.ai"
    res_reg = client.post("/api/auth/register", json={
        "email": test_direct_email,
        "password": "Password123!"
    })
    assert res_reg.status_code == 200, f"Registration failed: {res_reg.text}"
    reg_json = res_reg.json()
    assert reg_json["success"] is True
    assert reg_json["user"]["email"] == test_direct_email
    print("   ✅ Direct registration endpoint verified.")

    print("🎉 All Server Auth tests passed successfully!")

if __name__ == "__main__":
    test_auth_endpoints()
