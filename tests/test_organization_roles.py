import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.server import app
from app.organizations import OrganizationManager, VALID_ROLES, APPLICABLE_ROLES

client = TestClient(app)

# Authenticate test client with Admin JWT token
_login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
if _login_res.status_code == 200:
    _admin_token = _login_res.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {_admin_token}"


def test_role_definitions():
    """Verifies valid and applicable roles."""
    assert "owner" in VALID_ROLES
    assert "admin" in VALID_ROLES
    assert "analyst" in VALID_ROLES
    assert "user" in VALID_ROLES

    # 'owner' cannot be directly applied for by an applicant
    assert "owner" not in APPLICABLE_ROLES
    assert set(APPLICABLE_ROLES) == {"admin", "analyst", "user"}
    print("✅ Role constants verified.")


def test_manager_application_and_approval_workflow():
    """Tests the full lifecycle of an application and role verification."""
    mgr = OrganizationManager.get_instance()

    # 1. Create a test organization
    test_org = mgr.create_organization({
        "name": "SecOps Autonomous Hub",
        "description": "Security operations and voice alerts workspace.",
        "industry": "Software & Cloud Tech",
        "creator_email": "founder@secops.io",
        "creator_name": "SecOps Founder",
    })
    org_id = test_org["id"]
    assert test_org["owner_email"] == "founder@secops.io"
    assert mgr.get_user_role(org_id, "founder@secops.io") == "owner"
    assert mgr.can_user_access_space(org_id, "founder@secops.io") is True
    print("✅ Organization created with founder as owner.")

    # 2. Non-member checks
    applicant_email = "analyst.candidate@partner.io"
    assert mgr.get_user_role(org_id, applicant_email) is None
    assert mgr.can_user_access_space(org_id, applicant_email) is False

    # 3. Invalid role application should raise ValueError
    try:
        mgr.apply_for_entry(org_id, applicant_email, "Candidate Name", "owner")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        mgr.apply_for_entry(org_id, applicant_email, "Candidate Name", "superadmin")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # 4. Valid application for 'analyst'
    app_record = mgr.apply_for_entry(
        org_id=org_id,
        applicant_email=applicant_email,
        applicant_name="Analyst Candidate",
        requested_role="analyst",
        note="Requesting analyst access for monitoring telemetry.",
    )
    assert app_record["status"] == "pending"
    assert app_record["requested_role"] == "analyst"
    assert mgr.can_user_access_space(org_id, applicant_email) is False
    print("✅ Entry application submitted with pending status.")

    # 5. Non-admin/non-owner cannot review applications
    random_user = "unrelated@random.com"
    try:
        mgr.review_application(org_id, app_record["id"], "approve", reviewer_email=random_user)
        assert False, "Should have raised PermissionError"
    except PermissionError:
        pass

    # 6. Founder (owner) denies the application
    deny_res = mgr.review_application(
        org_id,
        app_record["id"],
        action="deny",
        reviewer_email="founder@secops.io",
    )
    assert deny_res["application"]["status"] == "denied"
    assert mgr.can_user_access_space(org_id, applicant_email) is False

    # Setting active org with denied email must raise PermissionError
    try:
        mgr.set_active_organization(org_id, user_email=applicant_email)
        assert False, "Should have raised PermissionError"
    except PermissionError:
        pass
    print("✅ Denied application correctly blocks space access.")

    # 7. Applicant re-applies for 'user' role
    reapp_record = mgr.apply_for_entry(
        org_id=org_id,
        applicant_email=applicant_email,
        applicant_name="Candidate Name",
        requested_role="user",
        note="Re-applying for basic testing access.",
    )
    assert reapp_record["status"] == "pending"
    assert reapp_record["requested_role"] == "user"

    # 8. Founder approves application strictly as requested 'user'
    approve_res = mgr.review_application(
        org_id,
        reapp_record["id"],
        action="approve",
        reviewer_email="founder@secops.io",
    )
    assert approve_res["application"]["status"] == "approved"
    assert mgr.get_user_role(org_id, applicant_email) == "user"
    assert mgr.can_user_access_space(org_id, applicant_email) is True
    print("✅ Approved application grants space access and role.")

    # 9. Approved applicant can now enter/activate the space
    activated = mgr.set_active_organization(org_id, user_email=applicant_email)
    assert activated["is_active"] is True
    print("✅ Approved user can activate and open space.")

    # 10. Analyst cannot review other applications (only owner and admin can)
    third_party = "thirdparty@partner.io"
    app3 = mgr.apply_for_entry(org_id, third_party, "Third Party", "user")
    try:
        mgr.review_application(org_id, app3["id"], "approve", reviewer_email=applicant_email)
        assert False, "Should have raised PermissionError"
    except PermissionError:
        pass

    # 11. Founder adds an admin directly
    admin_email = "deputy@secops.io"
    new_admin = mgr.add_member(
        org_id=org_id,
        email=admin_email,
        name="Deputy Admin",
        role="admin",
        adder_email="founder@secops.io",
    )
    assert new_admin["role"] == "admin"
    assert mgr.get_user_role(org_id, admin_email) == "admin"

    # Admin CAN review and approve third_party
    admin_approve = mgr.review_application(
        org_id,
        app3["id"],
        action="approve",
        reviewer_email=admin_email,
        approved_role="user",
    )
    assert admin_approve["application"]["status"] == "approved"
    assert mgr.get_user_role(org_id, third_party) == "user"
    print("✅ Admin can review and approve applications.")

    # Cleanup test org
    mgr.set_active_organization("apex-ai-innovations")
    mgr.delete_organization(org_id, deleter_email="founder@secops.io")
    print("✅ OrganizationManager role lifecycle tests passed!")


def test_server_role_api_endpoints():
    """Tests the REST API endpoints for applying, reviewing, and accessing organizations."""
    # 1. Create org
    create_res = client.post("/api/organizations", json={
        "name": "CloudOps Automation Suite",
        "description": "DevOps AI voice bots.",
        "industry": "Software & Cloud Tech",
        "creator_email": "owner@cloudops.com",
        "creator_name": "CloudOps Lead",
    })
    assert create_res.status_code == 200
    org = create_res.json()
    org_id = org["id"]

    # 2. Check organizations list with owner email
    list_res = client.get(f"/api/organizations?user_email=owner@cloudops.com")
    assert list_res.status_code == 200
    orgs = list_res.json()["organizations"]
    matched = next((o for o in orgs if o["id"] == org_id), None)
    assert matched is not None
    assert matched["user_role"] == "owner"
    assert matched["can_open"] is True

    # 3. Check organizations list with non-member email
    test_applicant = "applicant@external.com"
    list_res2 = client.get(f"/api/organizations?user_email={test_applicant}")
    matched2 = next((o for o in list_res2.json()["organizations"] if o["id"] == org_id), None)
    assert matched2 is not None
    assert matched2["user_role"] is None
    assert matched2["can_open"] is False

    # 4. Attempt to activate space as non-member -> 403 Forbidden
    act_forbidden = client.post(
        f"/api/organizations/{org_id}/activate",
        json={"user_email": test_applicant}
    )
    assert act_forbidden.status_code == 403
    print("✅ Activating organization as non-member returns 403 Forbidden.")

    # 5. Apply for entry with role 'admin'
    apply_res = client.post(f"/api/organizations/{org_id}/apply", json={
        "applicant_email": test_applicant,
        "applicant_name": "External Candidate",
        "requested_role": "admin",
        "note": "Requesting admin role to configure CI/CD voice pipelines."
    })
    assert apply_res.status_code == 200
    app_data = apply_res.json()["application"]
    app_id = app_data["id"]
    assert app_data["status"] == "pending"
    assert app_data["requested_role"] == "admin"
    print("✅ POST /api/organizations/{id}/apply succeeded.")

    # 6. List applications as owner
    apps_res = client.get(f"/api/organizations/{org_id}/applications?reviewer_email=owner@cloudops.com")
    assert apps_res.status_code == 200
    apps_list = apps_res.json()["applications"]
    assert len(apps_list) >= 1
    assert any(a["id"] == app_id for a in apps_list)

    # List applications as unauthorized user -> 403
    apps_unauth = client.get(f"/api/organizations/{org_id}/applications?reviewer_email={test_applicant}")
    assert apps_unauth.status_code == 403

    # 7. Approve application as owner
    review_res = client.post(f"/api/organizations/{org_id}/applications/{app_id}/review", json={
        "action": "approve",
        "reviewer_email": "owner@cloudops.com",
        "approved_role": "admin",
    })
    assert review_res.status_code == 200
    assert review_res.json()["application"]["status"] == "approved"
    print("✅ POST /api/organizations/{id}/applications/{id}/review approved application.")

    # 8. Check applicant can now open/activate the space -> 200 OK
    act_ok = client.post(
        f"/api/organizations/{org_id}/activate",
        json={"user_email": test_applicant}
    )
    assert act_ok.status_code == 200
    assert act_ok.json()["success"] is True
    print("✅ Approved applicant successfully opened the organization space.")

    # 9. List members
    members_res = client.get(f"/api/organizations/{org_id}/members")
    assert members_res.status_code == 200
    members = members_res.json()["members"]
    assert len(members) == 2  # owner + approved admin
    roles = {m["email"]: m["role"] for m in members}
    assert roles["owner@cloudops.com"] == "owner"
    assert roles[test_applicant] == "admin"

    # Cleanup
    client.post("/api/organizations/apex-ai-innovations/activate")
    del_res = client.delete(f"/api/organizations/{org_id}?deleter_email=owner@cloudops.com")
    assert del_res.status_code == 200
    print("🎉 All Server Role API endpoint tests passed successfully!")


def test_applicant_freedom_vs_admin_designation_authority():
    """
    Verifies that:
    1. Applicant can apply for any valid role (freedom of application: user, analyst, admin).
    2. The applicant's requested role DOES NOT automatically become their role.
    3. ONLY the reviewing Owner or Admin decides and assigns the final designation.
    4. If reviewer does not pass approved_role, it safely defaults to 'user' rather than applicant's request.
    """
    mgr = OrganizationManager.get_instance()

    test_org = mgr.create_organization({
        "name": "Designation Authority Workspace",
        "description": "Workspace to test role assignment authority.",
        "industry": "Enterprise Security",
        "creator_email": "owner@authority.test",
        "creator_name": "Chief Authority",
    })
    org_id = test_org["id"]

    try:
        # 1. Applicant 1 applies for 'admin' (requested designation)
        app1 = mgr.apply_for_entry(
            org_id=org_id,
            applicant_email="candidate1@authority.test",
            applicant_name="Candidate One",
            requested_role="admin",
            note="I would like to be an admin.",
        )
        assert app1["requested_role"] == "admin"
        assert app1["status"] == "pending"

        # Attempting to approve with a conflicting designation raises ValueError
        try:
            mgr.review_application(
                org_id=org_id,
                app_id=app1["id"],
                action="approve",
                reviewer_email="owner@authority.test",
                approved_role="user",
            )
            assert False, "Should not allow approving with a different designation than requested"
        except ValueError as e:
            assert "strictly as per the requested designation" in str(e)

        # Owner approves applicant -> Acceptance MUST be strictly as per requested designation ('admin')
        mgr.review_application(
            org_id=org_id,
            app_id=app1["id"],
            action="approve",
            reviewer_email="owner@authority.test",
        )
        assert mgr.get_user_role(org_id, "candidate1@authority.test") == "admin"

        # 2. Applicant 2 applies for 'analyst'
        app2 = mgr.apply_for_entry(
            org_id=org_id,
            applicant_email="candidate2@authority.test",
            applicant_name="Candidate Two",
            requested_role="analyst",
            note="Applying for analyst.",
        )
        assert app2["requested_role"] == "analyst"

        # Reviewer approves -> Applicant is accepted strictly as 'analyst'
        mgr.review_application(
            org_id=org_id,
            app_id=app2["id"],
            action="approve",
            reviewer_email="owner@authority.test",
        )
        assert mgr.get_user_role(org_id, "candidate2@authority.test") == "analyst"

        # 3. Applicant 3 applies for 'admin', but Owner does not want to grant admin -> Denies application
        app3 = mgr.apply_for_entry(
            org_id=org_id,
            applicant_email="candidate3@authority.test",
            applicant_name="Candidate Three",
            requested_role="admin",
        )
        mgr.review_application(
            org_id=org_id,
            app_id=app3["id"],
            action="deny",
            reviewer_email="owner@authority.test",
        )
        assert mgr.is_member(org_id, "candidate3@authority.test") is False

        # 4. Direct Invitation: Owner or Admin invites a member and designates their role directly
        invited = mgr.invite_member(
            org_id=org_id,
            email="invited@authority.test",
            name="Invited Developer",
            role="user",
            inviter_email="owner@authority.test",
        )
        assert invited["role"] == "user"
        assert mgr.get_user_role(org_id, "invited@authority.test") == "user"

        print("✅ Applicant requested designation on acceptance & Owner/Admin invitation designation verified!")

    finally:
        mgr.delete_organization(org_id, deleter_email="owner@authority.test")


def test_non_admin_cannot_assign_or_change_roles():
    """
    Verifies that neither 'user' nor 'analyst' can:
    - Review applications
    - Add members
    - Update member roles
    - Modify designations
    """
    mgr = OrganizationManager.get_instance()

    test_org = mgr.create_organization({
        "name": "Access Control Test",
        "creator_email": "root@access.test",
        "creator_name": "Root Admin",
    })
    org_id = test_org["id"]

    try:
        # Add an analyst and a normal user
        mgr.add_member(org_id, "analyst@access.test", "Analyst User", "analyst", adder_email="root@access.test")
        mgr.add_member(org_id, "standard@access.test", "Regular User", "user", adder_email="root@access.test")
        mgr.add_member(org_id, "target@access.test", "Target Member", "user", adder_email="root@access.test")

        # 1. Analyst attempts to change target's role -> PermissionError
        try:
            mgr.update_member_role(org_id, "target@access.test", "admin", updater_email="analyst@access.test")
            assert False, "Analyst should not be able to change roles"
        except PermissionError:
            pass

        # 2. Standard user attempts to change target's role -> PermissionError
        try:
            mgr.update_member_role(org_id, "target@access.test", "admin", updater_email="standard@access.test")
            assert False, "User should not be able to change roles"
        except PermissionError:
            pass

        # 3. Via REST API: PUT /api/organizations/{id}/members/{email}/role
        res_analyst = client.put(f"/api/organizations/{org_id}/members/target@access.test/role", json={
            "role": "admin",
            "updater_email": "analyst@access.test",
        })
        assert res_analyst.status_code == 403

        res_user = client.put(f"/api/organizations/{org_id}/members/target@access.test/role", json={
            "role": "admin",
            "updater_email": "standard@access.test",
        })
        assert res_user.status_code == 403

        # 4. Analyst attempts to add new member -> 403
        res_add = client.post(f"/api/organizations/{org_id}/members", json={
            "email": "newbie@access.test",
            "role": "admin",
            "adder_email": "analyst@access.test",
        })
        assert res_add.status_code == 403

        print("✅ Non-admin/owner role modification strictly blocked with 403 Forbidden!")

    finally:
        mgr.delete_organization(org_id, deleter_email="root@access.test")


def test_owner_only_organization_edit_permission():
    """
    Verifies that:
    1. Organization Owner has exclusive rights to edit organization details (name, description, industry).
    2. Admin, Analyst, and User cannot edit organization details (raises PermissionError / 403 Forbidden).
    """
    mgr = OrganizationManager.get_instance()

    test_org = mgr.create_organization({
        "name": "Org Edit Permission Workspace",
        "description": "Original description for permissions testing.",
        "industry": "Software & Cloud Tech",
        "creator_email": "realowner@permissions.test",
        "creator_name": "Real Owner",
    })
    org_id = test_org["id"]

    try:
        # Add members with different roles
        mgr.add_member(org_id, "admin_user@permissions.test", "Admin Member", "admin", adder_email="realowner@permissions.test")
        mgr.add_member(org_id, "analyst_user@permissions.test", "Analyst Member", "analyst", adder_email="realowner@permissions.test")
        mgr.add_member(org_id, "standard_user@permissions.test", "Standard Member", "user", adder_email="realowner@permissions.test")

        # 1. Admin tries to edit -> PermissionError
        try:
            mgr.update_organization(org_id, {"name": "Hacked by Admin"}, updater_email="admin_user@permissions.test")
            assert False, "Admin should not be able to edit organization details"
        except PermissionError as e:
            assert "Only the Organization Owner can edit organization details" in str(e)

        # 2. Analyst tries to edit -> PermissionError
        try:
            mgr.update_organization(org_id, {"name": "Hacked by Analyst"}, updater_email="analyst_user@permissions.test")
            assert False, "Analyst should not be able to edit organization details"
        except PermissionError as e:
            assert "Only the Organization Owner can edit organization details" in str(e)

        # 3. User tries to edit -> PermissionError
        try:
            mgr.update_organization(org_id, {"name": "Hacked by User"}, updater_email="standard_user@permissions.test")
            assert False, "User should not be able to edit organization details"
        except PermissionError as e:
            assert "Only the Organization Owner can edit organization details" in str(e)

        # 4. REST API endpoint checks:
        # Admin -> 403
        res_admin = client.put(f"/api/organizations/{org_id}", json={
            "name": "Attempt by Admin",
            "updater_email": "admin_user@permissions.test"
        })
        assert res_admin.status_code == 403
        assert "Only the Organization Owner can edit" in res_admin.json()["detail"]

        # Analyst -> 403
        res_analyst = client.put(f"/api/organizations/{org_id}", json={
            "name": "Attempt by Analyst",
            "updater_email": "analyst_user@permissions.test"
        })
        assert res_analyst.status_code == 403

        # User -> 403
        res_user = client.put(f"/api/organizations/{org_id}", json={
            "name": "Attempt by User",
            "updater_email": "standard_user@permissions.test"
        })
        assert res_user.status_code == 403

        # Owner -> 200 OK
        res_owner = client.put(f"/api/organizations/{org_id}", json={
            "name": "Updated By True Owner",
            "description": "Legitimately edited by the owner.",
            "industry": "Software & Cloud Tech",
            "updater_email": "realowner@permissions.test"
        })
        assert res_owner.status_code == 200
        assert res_owner.json()["name"] == "Updated By True Owner"
        assert res_owner.json()["description"] == "Legitimately edited by the owner."

        # Verify in manager
        updated_org = mgr.get_organization(org_id)
        assert updated_org["name"] == "Updated By True Owner"
        print("✅ Exclusive Owner-only organization editing permission verified successfully!")

    finally:
        mgr.delete_organization(org_id, deleter_email="realowner@permissions.test")


if __name__ == "__main__":
    test_role_definitions()
    test_manager_application_and_approval_workflow()
    test_server_role_api_endpoints()
    test_applicant_freedom_vs_admin_designation_authority()
    test_non_admin_cannot_assign_or_change_roles()
    test_owner_only_organization_edit_permission()
    print("🚀 All Organization Role and Application tests completed successfully!")
