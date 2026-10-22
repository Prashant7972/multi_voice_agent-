"""
Unit tests for UserManager and EmailService in app/auth.py & app/email_service.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import UserManager
from app.email_service import EmailService

def test_user_manager_and_otp():
    print("🧪 Testing UserManager and OTP Engine...")
    mgr = UserManager.get_instance()
    email_svc = EmailService.get_instance()

    # 1. OTP Generation and Verification
    test_email = "tester-otp@apexai.io"
    otp = mgr.generate_otp(test_email, purpose="registration")
    assert len(otp) == 6 and otp.isdigit(), f"Invalid OTP format: {otp}"
    print(f"   ✅ OTP generated: {otp}")

    # Email Service Send test
    send_result = email_svc.send_otp(test_email, otp, purpose="registration")
    assert send_result["success"] is True
    print(f"   ✅ Email Service delivered OTP: {send_result}")

    # Verify with wrong OTP
    assert not mgr.verify_otp(test_email, "000000", purpose="registration")
    print("   ✅ Wrong OTP rejected.")

    # Verify with correct OTP
    assert mgr.verify_otp(test_email, otp, purpose="registration")
    print("   ✅ Correct OTP verified.")

    # Verify single-use (cannot reuse)
    assert not mgr.verify_otp(test_email, otp, purpose="registration")
    print("   ✅ Reused OTP rejected (single-use enforced).")

    # 2. Test Password Reset
    admin_user = mgr.get_user_by_email("admin@apexai.io")
    assert admin_user is not None
    orig_auth = mgr.authenticate("admin@apexai.io", "admin123")
    assert orig_auth is not None

    # Reset password
    success = mgr.reset_password("admin@apexai.io", "new_admin_password_456")
    assert success is True
    assert mgr.authenticate("admin@apexai.io", "new_admin_password_456") is not None
    assert mgr.authenticate("admin@apexai.io", "admin123") is None
    print("   ✅ Password reset verified.")

    # Restore original password for admin
    mgr.reset_password("admin@apexai.io", "admin123")
    assert mgr.authenticate("admin@apexai.io", "admin123") is not None
    print("   ✅ Restored original admin password.")

    print("🎉 All UserManager & EmailService tests passed successfully!")

if __name__ == "__main__":
    test_user_manager_and_otp()
