import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.server import app, org_manager

client = TestClient(app)

# Authenticate test client with Admin JWT token
_login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
_admin_token = _login_res.json()["access_token"]
client.headers["Authorization"] = f"Bearer {_admin_token}"



def test_get_organizations():
    res = client.get("/api/organizations")
    assert res.status_code == 200
    data = res.json()
    assert "organizations" in data
    assert "active_organization" in data
    assert len(data["organizations"]) >= 1
    assert data["active_organization"] is not None
    assert "name" in data["active_organization"]
    print("✅ GET /api/organizations passed")


def test_organization_crud_lifecycle():
    # 1. Create organization
    test_org_payload = {
        "name": "Test Neural Health AI",
        "industry": "Healthcare & Telehealth",
        "description": "Medical triage and telehealth voice automation systems."
    }
    create_res = client.post("/api/organizations", json=test_org_payload)
    assert create_res.status_code == 200
    created = create_res.json()
    org_id = created["id"]
    assert created["name"] == "Test Neural Health AI"
    assert created["industry"] == "Healthcare & Telehealth"
    print(f"✅ POST /api/organizations passed (ID: {org_id})")

    # 2. Get organization detail
    detail_res = client.get(f"/api/organizations/{org_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["name"] == "Test Neural Health AI"
    print("✅ GET /api/organizations/{id} passed")

    # 3. Update organization
    update_res = client.put(f"/api/organizations/{org_id}", json={
        "name": "Neural Health AI Global",
        "industry": "Healthcare & Telehealth",
        "description": "Updated global telehealth operations."
    })
    assert update_res.status_code == 200
    assert update_res.json()["name"] == "Neural Health AI Global"
    print("✅ PUT /api/organizations/{id} passed")

    # 4. Activate organization
    activate_res = client.post(f"/api/organizations/{org_id}/activate")
    assert activate_res.status_code == 200
    assert activate_res.json()["active_organization"]["id"] == org_id
    print("✅ POST /api/organizations/{id}/activate passed")

    # 5. Check agents scoped with active organization
    agents_res = client.get("/api/agents")
    assert agents_res.status_code == 200
    agents_data = agents_res.json()
    assert agents_data["active_organization"]["id"] == org_id
    print("✅ GET /api/agents with active_organization passed")

    # 6. Delete test organization
    # First re-activate the default org so we can safely delete the test org
    client.post("/api/organizations/apex-ai-innovations/activate")
    del_res = client.delete(f"/api/organizations/{org_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True
    print("✅ DELETE /api/organizations/{id} passed")


def test_index_html_renders_organization_elements():
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "Organization Overview" in html
    assert 'class="org-hero-card"' not in html
    assert "orgSectionSettings" in html
    assert "navActiveOrgName" in html
    assert "viewOrg" in html
    assert "orgModal" in html
    assert "Apex AI Innovations" in html
    assert "viewAuthDashboard" in html
    assert "loginEmail" in html
    assert "loginPassword" in html
    assert "btnAuthSubmit" in html
    assert "btn-topbar-switch-group" in html
    assert "btn-topbar-switch-main" in html
    assert "btn-topbar-switch-arrow" in html
    assert "handleSwitchOrgClick" in html
    assert "navigateInitialUserDashboard" in html
    assert "last_active_org_id" in html
    assert "sidebarNavSettings" in html
    assert "sidebarNavSettingsEmail" in html
    assert "btnSubnavSettings" in html
    assert "orgSectionSettings" in html
    assert "settingsViewNotifications" in html
    assert "settingsViewEditOrg" in html
    assert "btnSettingsSaveOrg" in html
    assert 'id="btnSettingsSaveOrg" class="btn-org-card-primary" style="display: none;' in html
    assert "settingsViewInvite" in html
    assert "settingsInviteEmail" in html
    assert "settingsInviteRole" in html
    assert "loadOrgSettingsView" in html
    assert "renderOrganizations" in html
    assert 'id="tabSettingsOrg"' in html
    assert "function isCurrentUserOrgOwner(" in html
    # tabSettingsOrg must be hidden by default (display: none;) so only verified owners see it
    tab_org_idx = html.find('id="tabSettingsOrg"')
    assert "display: none;" in html[tab_org_idx:tab_org_idx + 260]
    print("✅ UI HTML contains Organization Overview, Settings Section, Active Email Setting Item, Hidden Edit Button, Invite Panel, and Filtered Switch Dropdown")


def test_settings_workflow_and_permissions():
    """
    Validates backend API behavior powering the Settings Section:
    1. Notification card: only owner/admin can list applications and review them.
    2. Edit card: only owner can edit organization details.
    """
    # 1. Create a test organization
    create_res = client.post("/api/organizations", json={
        "name": "Settings Test Workspace",
        "industry": "Technology & Software Services",
        "description": "Workspace for settings permission validation.",
        "creator_email": "owner@settings.test",
        "creator_name": "Settings Owner",
    })
    assert create_res.status_code == 200
    org_id = create_res.json()["id"]

    try:
        # Add an admin and a regular user
        client.post(f"/api/organizations/{org_id}/members", json={
            "email": "admin@settings.test",
            "name": "Settings Admin",
            "role": "admin",
            "adder_email": "owner@settings.test",
        })
        client.post(f"/api/organizations/{org_id}/members", json={
            "email": "user@settings.test",
            "name": "Settings User",
            "role": "user",
            "adder_email": "owner@settings.test",
        })

        # 2. Submit applicant request
        apply_res = client.post(f"/api/organizations/{org_id}/apply", json={
            "applicant_email": "applicant@settings.test",
            "applicant_name": "Applicant",
            "requested_role": "analyst",
            "note": "Looking for telemetry access."
        })
        assert apply_res.status_code == 200
        app_id = apply_res.json()["application"]["id"]

        # 3. Notification Card Access:
        # User -> 403 Forbidden
        user_apps_res = client.get(f"/api/organizations/{org_id}/applications?reviewer_email=user@settings.test")
        assert user_apps_res.status_code == 403

        # Admin -> 200 OK
        admin_apps_res = client.get(f"/api/organizations/{org_id}/applications?reviewer_email=admin@settings.test")
        assert admin_apps_res.status_code == 200
        assert any(a["id"] == app_id for a in admin_apps_res.json()["applications"])

        # Owner -> 200 OK
        owner_apps_res = client.get(f"/api/organizations/{org_id}/applications?reviewer_email=owner@settings.test")
        assert owner_apps_res.status_code == 200

        # 4. Review & Approval:
        # Conflicting role designation is rejected:
        conflict_res = client.post(f"/api/organizations/{org_id}/applications/{app_id}/review", json={
            "action": "approve",
            "reviewer_email": "admin@settings.test",
            "approved_role": "user"
        })
        assert conflict_res.status_code == 400
        assert "strictly as per the requested designation" in conflict_res.json()["detail"]

        # Admin approves applicant strictly as requested ('analyst')
        review_res = client.post(f"/api/organizations/{org_id}/applications/{app_id}/review", json={
            "action": "approve",
            "reviewer_email": "admin@settings.test",
        })
        assert review_res.status_code == 200
        assert review_res.json()["application"]["status"] == "approved"
        assert review_res.json()["application"]["approved_role"] == "analyst"

        # 5. Direct Invitation with Designation:
        # Non-member/regular user cannot invite -> 403
        unauthorized_invite = client.post(f"/api/organizations/{org_id}/invite", json={
            "email": "invitee@settings.test",
            "role": "analyst",
            "inviter_email": "user@settings.test",
        })
        assert unauthorized_invite.status_code == 403

        # Admin or Owner invites with explicit designation -> 200
        invite_res = client.post(f"/api/organizations/{org_id}/invite", json={
            "email": "invited.colleague@settings.test",
            "role": "analyst",
            "name": "Invited Colleague",
            "inviter_email": "admin@settings.test",
        })
        assert invite_res.status_code == 200
        assert invite_res.json()["member"]["role"] == "analyst"
        assert invite_res.json()["success"] is True

        # 6. Edit Organization Card Access:
        # Admin -> 403 Forbidden
        admin_edit_res = client.put(f"/api/organizations/{org_id}", json={
            "name": "Admin Edited Title",
            "updater_email": "admin@settings.test"
        })
        assert admin_edit_res.status_code == 403

        # User -> 403 Forbidden
        user_edit_res = client.put(f"/api/organizations/{org_id}", json={
            "name": "User Edited Title",
            "updater_email": "user@settings.test"
        })
        assert user_edit_res.status_code == 403

        # Owner -> 200 OK
        owner_edit_res = client.put(f"/api/organizations/{org_id}", json={
            "name": "Owner Updated Workspace",
            "description": "Legitimately edited from Settings Section by owner.",
            "industry": "Technology & Software Services",
            "updater_email": "owner@settings.test"
        })
        assert owner_edit_res.status_code == 200
        assert owner_edit_res.json()["name"] == "Owner Updated Workspace"
        print("✅ Settings workflow permissions (notifications for admin/owner, edit for owner only) verified!")

    finally:
        client.delete(f"/api/organizations/{org_id}?deleter_email=owner@settings.test")


def test_new_user_organization_visibility_and_apply_flow():
    """
    Validates:
    1. A new user or non-member receives all registered organizations via GET /api/organizations?user_email=...
    2. Organizations where the user is not a member have can_open == False and user_role == None.
    3. Attempting to activate an organization as a non-member returns 403 Forbidden.
    4. The user can apply to join with designation ('analyst') and optional note via POST /api/organizations/{org_id}/apply.
    5. After applying, GET /api/organizations?user_email=... returns user_application with status 'pending' and requested_role 'analyst'.
    6. HTML includes renderOrganizations, applyOrgModal, openApplyOrgModal, handleApplyOrgSubmit, and clean OTP layout.
    """
    new_user_email = "newapplicant@example.com"
    apex_org_raw = org_manager.get_organization("apex-ai-innovations")
    if apex_org_raw:
        apex_org_raw["members"] = [m for m in apex_org_raw.get("members", []) if m.get("email") != new_user_email]
        apex_org_raw["applications"] = [a for a in apex_org_raw.get("applications", []) if a.get("applicant_email") != new_user_email]
        org_manager.save()

    # 1. New user lists organizations
    res = client.get(f"/api/organizations?user_email={new_user_email}")
    assert res.status_code == 200
    data = res.json()
    orgs = data["organizations"]
    assert len(orgs) >= 1

    # Apex AI Innovations default org
    apex_org = [o for o in orgs if o["id"] == "apex-ai-innovations"][0]
    assert apex_org["can_open"] is False
    assert apex_org["user_role"] is None

    # 2. Non-member cannot activate
    activate_res = client.post(f"/api/organizations/apex-ai-innovations/activate", json={"user_email": new_user_email})
    assert activate_res.status_code == 403
    assert "not an approved member" in activate_res.json()["detail"]

    # 3. Non-member applies to join as analyst
    apply_payload = {
        "applicant_email": new_user_email,
        "applicant_name": "New Applicant",
        "requested_role": "analyst",
        "note": "Requesting analyst access to view telemetry."
    }
    apply_res = client.post(f"/api/organizations/apex-ai-innovations/apply", json=apply_payload)
    assert apply_res.status_code == 200
    app_data = apply_res.json()
    assert app_data["success"] is True
    assert app_data["application"]["requested_role"] == "analyst"
    assert app_data["application"]["status"] == "pending"

    # 4. Check list again: now user_application is pending
    res_after = client.get(f"/api/organizations?user_email={new_user_email}")
    assert res_after.status_code == 200
    apex_after = [o for o in res_after.json()["organizations"] if o["id"] == "apex-ai-innovations"][0]
    assert apex_after["can_open"] is False
    assert apex_after["user_application"] is not None
    assert apex_after["user_application"]["status"] == "pending"
    assert apex_after["user_application"]["requested_role"] == "analyst"

    # 5. Check index.html elements
    index_res = client.get("/")
    assert index_res.status_code == 200
    html = index_res.text
    assert "function renderOrganizations(" in html
    assert "openApplyOrgModal" in html
    assert "applyOrgModal" in html
    assert "handleApplyOrgSubmit" in html
    assert "activateAndOpenOrg" in html
    assert "otp-input-field" in html
    assert "regStep2Form" in html
    print("✅ test_new_user_organization_visibility_and_apply_flow passed!")


if __name__ == "__main__":
    test_get_organizations()
    test_organization_crud_lifecycle()
    test_index_html_renders_organization_elements()
    test_settings_workflow_and_permissions()
    test_new_user_organization_visibility_and_apply_flow()
    print("🎉 All Server Organization tests passed successfully!")
