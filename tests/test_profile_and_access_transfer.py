import os
import pytest
from fastapi.testclient import TestClient

from app.server import app
from app.auth import UserManager
from app.organizations import OrganizationManager

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_state():
    """Ensure managers reload fresh state or have clean instances."""
    yield

import uuid

def test_profile_update_and_org_access_transfer():
    user_mgr = UserManager.get_instance()
    user_mgr.load()
    org_mgr = OrganizationManager.get_instance()
    org_mgr.load()

    uid = uuid.uuid4().hex[:6]
    test_email = f"alex.{uid}@example.com"
    test_pwd = "Password123!"
    test_name = "Alex Original"

    reg_res = client.post("/api/auth/register", json={
        "email": test_email,
        "password": test_pwd,
        "name": test_name
    })
    # If already registered, login
    if reg_res.status_code == 201:
        tokens = reg_res.json()
    else:
        login_res = client.post("/api/auth/login", json={
            "email": test_email,
            "password": test_pwd
        })
        assert login_res.status_code == 200
        tokens = login_res.json()

    access_token = tokens["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # 2. Create an organization owned by this user
    org_create_res = client.post(
        "/api/organizations",
        headers=headers,
        json={
            "name": f"Alex Ventures {uid}",
            "description": "Transfer testing organization",
            "tier": "enterprise"
        }
    )
    assert org_create_res.status_code in [200, 201]
    org_data = org_create_res.json()
    org_id = org_data["id"]
    assert org_data["owner_email"] == test_email

    # 3. Test wrong current password when trying to update password
    fail_pwd_res = client.put(
        "/api/auth/profile",
        headers=headers,
        json={
            "name": test_name,
            "email": test_email,
            "current_password": "WrongPassword999!",
            "new_password": "NewSecretPassword456!"
        }
    )
    assert fail_pwd_res.status_code == 400
    assert "Current password does not match" in fail_pwd_res.json()["detail"]

    # 4. Successfully update Profile: Name, Avatar, and Email
    new_email = f"alex.promoted.{uid}@newdomain.com"
    new_name = "Alex Senior Architect"
    new_avatar = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    update_res = client.put(
        "/api/auth/profile",
        headers=headers,
        json={
            "name": new_name,
            "email": new_email,
            "avatar_url": new_avatar,
            "current_password": test_pwd,
            "new_password": "NewSecretPassword456!"
        }
    )
    assert update_res.status_code == 200, f"Update profile failed: {update_res.text}"
    update_data = update_res.json()
    assert update_data["user"]["name"] == new_name
    assert update_data["user"]["email"] == new_email
    assert update_data["user"]["avatar_url"] == new_avatar
    assert "access_token" in update_data

    # Use the fresh token issued with new claims
    new_token = update_data["access_token"]
    new_headers = {"Authorization": f"Bearer {new_token}"}

    # 5. Verify that Organization ownership automatically transferred to the new email
    org_check_res = client.get(f"/api/organizations/{org_id}", headers=new_headers)
    assert org_check_res.status_code == 200
    updated_org = org_check_res.json()
    assert updated_org["owner_email"] == new_email, f"Expected owner_email to be {new_email}, got {updated_org['owner_email']}"

    # Verify user role in org is still owner
    user_role = org_mgr.get_user_role(org_id, new_email)
    assert user_role == "owner", f"User role should remain owner, got {user_role}"

    # Verify old email no longer has access or ownership
    old_role = org_mgr.get_user_role(org_id, test_email)
    assert old_role != "owner"

    # 6. Verify login with the updated password and updated email works
    relogin_res = client.post("/api/auth/login", json={
        "email": new_email,
        "password": "NewSecretPassword456!"
    })
    assert relogin_res.status_code == 200
    assert relogin_res.json()["user"]["name"] == new_name
    assert relogin_res.json()["user"]["email"] == new_email

    # Verify old email and old password can no longer login
    old_login_res = client.post("/api/auth/login", json={
        "email": test_email,
        "password": test_pwd
    })
    assert old_login_res.status_code in [400, 401]

def test_profile_duplicate_email_conflict():
    # Attempting to change email to an already registered email must return 400
    uid = uuid.uuid4().hex[:6]
    email_temp = f"temp.conflict.{uid}@example.com"
    pwd = "TempPassword123!"
    reg = client.post("/api/auth/register", json={
        "email": email_temp,
        "password": pwd,
        "name": "Temp Conflict User"
    })
    if reg.status_code == 201:
        token = reg.json()["access_token"]
    else:
        token = client.post("/api/auth/login", json={"email": email_temp, "password": pwd}).json()["access_token"]

    res = client.put(
        "/api/auth/profile",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "Temp Conflict User",
            "email": "admin@apexai.io"  # Collision with existing admin
        }
    )
    assert res.status_code == 400
    assert "already in use" in res.json()["detail"].lower()

def test_avatar_upload_endpoint():
    # Login as admin
    login_res = client.post("/api/auth/login", json={
        "email": "admin@apexai.io",
        "password": "admin123"
    })
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Post avatar via data URL
    data_url = "data:image/svg+xml;utf8,<svg></svg>"
    res = client.post(
        "/api/auth/avatar",
        headers=headers,
        json={"avatar_url": data_url}
    )
    assert res.status_code == 200
    assert res.json()["avatar_url"] == data_url
    assert res.json()["user"]["avatar_url"] == data_url
