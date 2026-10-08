"""
Integration tests for OTP Registration, Forgot Password, and Double Confirmation.
"""
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.server import app
from app.auth import UserManager

client = TestClient(app)

def test_otp_registration_and_reset_endpoints():
    print("🧪 Testing Server OTP Registration & Password Reset Endpoints...")
    mgr = UserManager.get_instance()

    test_user_email = "alex-newuser@apexai.io"

    # Clean up if existed previously
    mgr.users = [u for u in mgr.users if u.get("email") != test_user_email]
    mgr.save()

    # 1. Send OTP for registration
    res_send = client.post("/api/auth/otp/send", json={
        "email": test_user_email,
        "purpose": "registration"
    })
    assert res_send.status_code == 200, f"Failed to send registration OTP: {res_send.text}"
    send_data = res_send.json()
    assert send_data["success"] is True
    otp_code = send_data.get("dev_otp") or mgr.otps.get(f"{test_user_email}:registration", {}).get("code")
    assert otp_code is not None and len(otp_code) == 6
    print(f"   ✅ POST /api/auth/otp/send for registration passed (OTP: {otp_code})")

    # 2. Try registration with mismatched passwords -> should fail
    res_mismatch = client.post("/api/auth/register-with-otp", json={
        "email": test_user_email,
        "otp": otp_code,
        "password": "Password123!",
        "confirm_password": "DifferentPassword456!",
        "name": "Alex Mercer"
    })
    assert res_mismatch.status_code == 400
    assert "match" in res_mismatch.json()["detail"].lower()
    print("   ✅ Mismatched passwords rejected with 400 Bad Request.")

    # 3. Register with correct matching passwords and valid OTP
    res_reg = client.post("/api/auth/register-with-otp", json={
        "email": test_user_email,
        "otp": otp_code,
        "password": "Password123!",
        "confirm_password": "Password123!",
        "name": "Alex Mercer"
    })
    assert res_reg.status_code == 200, f"Registration failed: {res_reg.text}"
    reg_data = res_reg.json()
    assert reg_data["success"] is True
    assert reg_data["user"]["email"] == test_user_email
    assert "token" in reg_data
    print("   ✅ Successful registration with OTP verified!")

    # 4. Verify login with the newly created password
    res_login = client.post("/api/auth/login", json={
        "email": test_user_email,
        "password": "Password123!"
    })
    assert res_login.status_code == 200
    print("   ✅ New user successfully logged in with created password.")

    # 5. Forgot Password Flow
    # Request OTP for password reset
    res_reset_otp = client.post("/api/auth/otp/send", json={
        "email": test_user_email,
        "purpose": "password_reset"
    })
    assert res_reset_otp.status_code == 200
    reset_otp = res_reset_otp.json().get("dev_otp") or mgr.otps.get(f"{test_user_email}:password_reset", {}).get("code")
    assert reset_otp is not None
    print(f"   ✅ POST /api/auth/otp/send for password reset passed (OTP: {reset_otp})")

    # Try reset with mismatched passwords -> should fail
    res_reset_mismatch = client.post("/api/auth/reset-password", json={
        "email": test_user_email,
        "otp": reset_otp,
        "new_password": "NewSecretPassword789!",
        "confirm_password": "TypoInConfirmPassword!"
    })
    assert res_reset_mismatch.status_code == 400
    print("   ✅ Reset rejected on mismatched passwords.")

    # Reset with valid OTP and matching passwords
    res_reset = client.post("/api/auth/reset-password", json={
        "email": test_user_email,
        "otp": reset_otp,
        "new_password": "NewSecretPassword789!",
        "confirm_password": "NewSecretPassword789!"
    })
    assert res_reset.status_code == 200, f"Password reset failed: {res_reset.text}"
    print("   ✅ Password successfully reset via OTP.")

    # Old password should now fail
    res_old_login = client.post("/api/auth/login", json={
        "email": test_user_email,
        "password": "Password123!"
    })
    assert res_old_login.status_code == 401
    print("   ✅ Old password invalidated.")

    # New password should succeed
    res_new_login = client.post("/api/auth/login", json={
        "email": test_user_email,
        "password": "NewSecretPassword789!"
    })
    assert res_new_login.status_code == 200
    print("   ✅ Successfully authenticated with new confirmed password.")

    # Cleanup
    mgr.users = [u for u in mgr.users if u.get("email") != test_user_email]
    mgr.save()
    print("🎉 All Server OTP Registration & Password Reset tests passed successfully!")


def test_registration_without_otp_is_blocked():
    print("🧪 Testing that unverified registration without OTP is strictly blocked...")
    mgr = UserManager.get_instance()
    unverified_email = "blocked-no-otp@voicing.ai"
    mgr.users = [u for u in mgr.users if u.get("email") != unverified_email]
    mgr.save()

    # 1. Attempt POST /api/auth/register without OTP using browser user-agent
    res_browser = client.post(
        "/api/auth/register",
        json={"email": unverified_email, "password": "Password123!"},
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    assert res_browser.status_code == 400
    assert "otp" in res_browser.json()["detail"].lower()
    print("   ✅ POST /api/auth/register without OTP blocked for browser clients.")

    # 2. Attempt POST /api/auth/register with explicit OTP enforcement header
    res_enforce = client.post(
        "/api/auth/register",
        json={"email": unverified_email, "password": "Password123!"},
        headers={"x-enforce-otp": "true"}
    )
    assert res_enforce.status_code == 400
    assert "otp" in res_enforce.json()["detail"].lower()
    print("   ✅ POST /api/auth/register without OTP blocked with x-enforce-otp.")

    # 3. Attempt POST /api/auth/register with invalid OTP
    res_bad_otp = client.post(
        "/api/auth/register",
        json={"email": unverified_email, "password": "Password123!", "otp": "999999"},
        headers={"User-Agent": "Mozilla/5.0"}
    )
    assert res_bad_otp.status_code == 400
    assert "invalid" in res_bad_otp.json()["detail"].lower() or "expired" in res_bad_otp.json()["detail"].lower()
    print("   ✅ POST /api/auth/register with invalid OTP rejected.")

    # 4. Attempt POST /api/auth/register-with-otp with missing/invalid OTP
    res_missing_otp = client.post(
        "/api/auth/register-with-otp",
        json={
            "email": unverified_email,
            "password": "Password123!",
            "confirm_password": "Password123!",
            "otp": ""
        }
    )
    assert res_missing_otp.status_code == 400
    print("   ✅ POST /api/auth/register-with-otp with missing OTP rejected.")

    # Verify user was NOT created in database
    assert mgr.get_user_by_email(unverified_email) is None
    print("   ✅ Verified account was NOT created without valid OTP.")


def test_registration_ui_contains_strict_otp_elements():
    print("🧪 Testing UI template contains strictly OTP-verified registration flow...")
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    assert "regStep1Form" in html
    assert "handleSendRegOtp" in html
    assert "regStep2Form" in html
    assert "handleCompleteRegistration" in html
    assert "regOtp" in html
    assert "btnSendRegOtp" in html
    assert "btnCompleteReg" in html
    assert "btnDirectSignupSubmit" not in html
    assert "regDirectEmail" not in html
    assert "regDirectPassword" not in html
    print("   ✅ UI markup exclusively renders OTP-verified registration steps.")


if __name__ == "__main__":
    test_otp_registration_and_reset_endpoints()
    test_registration_without_otp_is_blocked()
    test_registration_ui_contains_strict_otp_elements()
