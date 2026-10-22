"""
Unit tests for UserManager in app/auth.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import UserManager

def test_user_manager():
    print("🧪 Testing UserManager...")
    mgr = UserManager.get_instance()
    
    # 1. Test authenticating admin
    admin = mgr.authenticate("admin@apexai.io", "admin123")
    assert admin is not None, "Failed to authenticate admin"
    assert admin["email"] == "admin@apexai.io"
    assert admin["role"] == "Organization Owner"
    assert "password_hash" not in admin, "Password hash must not be exposed"
    print("   ✅ Admin authentication verified.")

    # 2. Test authenticating dev
    dev = mgr.authenticate("dev@apexai.io", "dev123")
    assert dev is not None, "Failed to authenticate dev"
    assert dev["email"] == "dev@apexai.io"
    print("   ✅ Developer authentication verified.")

    # 3. Test invalid password
    invalid = mgr.authenticate("admin@apexai.io", "wrongpassword")
    assert invalid is None, "Invalid password should fail"
    print("   ✅ Invalid password rejected.")

    # 4. Test non-existent user
    non_user = mgr.authenticate("nobody@nowhere.com", "anypass")
    assert non_user is None, "Non-existent user should fail"
    print("   ✅ Non-existent user rejected.")

    # 5. Test registration
    new_email = "test-newuser@apexai.io"
    try:
        registered = mgr.register_user(new_email, "securepass123", name="New Test User", role="Tester")
        assert registered["email"] == new_email
        assert registered["name"] == "New Test User"
        assert mgr.authenticate(new_email, "securepass123") is not None
        print("   ✅ User registration verified.")
    finally:
        # Cleanup
        mgr.users = [u for u in mgr.users if u.get("email") != new_email]
        mgr.save()

    print("🎉 All UserManager tests passed successfully!")

if __name__ == "__main__":
    test_user_manager()
