import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)

# Authenticate test client with Admin JWT token
_login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
_admin_token = _login_res.json()["access_token"]
client.headers["Authorization"] = f"Bearer {_admin_token}"



def test_get_organization_analytics_endpoint():
    print("🧪 Testing GET /api/organizations/{org_id}/analytics endpoint...")
    res = client.get("/api/organizations/apex-ai-innovations/analytics")
    assert res.status_code == 200
    data = res.json()
    assert data["org_id"] == "apex-ai-innovations"
    assert "summary" in data
    assert "trend_7d" in data
    assert "channels" in data
    assert "agents_breakdown" in data
    assert "recent_activity" in data
    assert isinstance(data["agents_breakdown"], list)
    print("   ✅ Valid organization analytics returned 200 with all required sections.")


def test_get_nonexistent_organization_analytics():
    print("🧪 Testing 404 for nonexistent organization analytics...")
    res = client.get("/api/organizations/nonexistent-org-xyz/analytics")
    assert res.status_code == 404
    print("   ✅ Non-existent organization correctly returned 404.")


if __name__ == "__main__":
    test_get_organization_analytics_endpoint()
    test_get_nonexistent_organization_analytics()
    print("🎉 All Server Analytics endpoint tests passed!")
