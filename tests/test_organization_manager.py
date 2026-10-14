import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.organizations import OrganizationManager


def test_organization_crud():
    print("🧪 Testing OrganizationManager...")
    mgr = OrganizationManager.get_instance()

    # 1. List organizations
    orgs = mgr.list_organizations()
    print(f"   Found {len(orgs)} organizations.")
    assert len(orgs) >= 1, "Must have at least 1 default organization"

    active_org = mgr.get_active_organization()
    print(f"   Active Organization: {active_org['name']}")
    assert active_org is not None

    # 2. Create organization
    new_org = mgr.create_organization({
        "name": "Global Health AI",
        "description": "Medical and clinical voice assistants for patients.",
        "industry": "Healthcare"
    })
    print(f"   Created organization: {new_org['id']} ({new_org['name']})")
    assert new_org["id"].startswith("global-health-ai")

    # 3. Activate organization
    activated = mgr.set_active_organization(new_org["id"])
    assert activated["is_active"] is True
    assert mgr.get_active_organization()["id"] == new_org["id"]
    print("   ✅ Organization activation verified.")

    # 4. Update organization
    updated = mgr.update_organization(new_org["id"], {
        "description": "Updated medical voice assistance systems."
    })
    assert "Updated medical" in updated["description"]
    print("   ✅ Organization update verified.")

    # 5. Delete organization
    deleted = mgr.delete_organization(new_org["id"])
    assert deleted is True
    assert mgr.get_organization(new_org["id"]) is None
    print("   ✅ Organization deletion verified.")

    # Reset default organization to active
    mgr.set_active_organization("apex-ai-innovations")
    print("🎉 All OrganizationManager tests passed successfully!")


if __name__ == "__main__":
    test_organization_crud()