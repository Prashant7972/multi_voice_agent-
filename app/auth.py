"""
Authentication and User Management Module for Local Voice Agent.
Provides credential validation, session user management, and user persistence.
"""
import json
import hashlib
import time
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import jwt

from app.config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
)

USERS_FILE = Path(__file__).resolve().parent / "data" / "users.json"
SALT = "local_voice_agent_salt_2026"


def hash_password(password: str) -> str:
    """Hashes a password with salt using SHA-256."""
    return hashlib.sha256((SALT + password).encode("utf-8")).hexdigest()


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generates a signed JWT access token with standard expiration and claims."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({
        "exp": expire,
        "iat": now,
        "type": "access",
    })
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generates a signed JWT refresh token for renewing access tokens."""
    to_encode = {
        "sub": str(data.get("sub") or data.get("id")),
        "email": data.get("email"),
        "type": "refresh",
    }
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({
        "exp": expire,
        "iat": now,
    })
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str, expected_type: Optional[str] = "access") -> Dict[str, Any]:
    """
    Decodes and validates a JWT token.
    Raises jwt.ExpiredSignatureError, jwt.InvalidTokenError, or ValueError.
    """
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    if expected_type and payload.get("type") != expected_type:
        raise ValueError(f"Invalid token type: expected '{expected_type}', got '{payload.get('type')}'")
    return payload



class UserManager:
    """
    Singleton manager for user authentication and user accounts.
    """
    _instance: Optional["UserManager"] = None

    def __init__(self, storage_path: Path = USERS_FILE):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.users: List[Dict[str, Any]] = []
        self.otps: Dict[str, Dict[str, Any]] = {}
        self.load()

    @classmethod
    def get_instance(cls) -> "UserManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        """Loads users from JSON storage or seeds defaults."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.users = json.load(f)
            except Exception as e:
                print(f"⚠️ Error reading users file: {e}. Reinitializing.")
                self.users = []

        if not self.users:
            self.users = [
                {
                    "id": "user-admin",
                    "email": "admin@apexai.io",
                    "password_hash": hash_password("admin123"),
                    "name": "Platform Administrator",
                    "role": "Organization Owner",
                    "created_at": time.time(),
                },
                {
                    "id": "user-dev",
                    "email": "dev@apexai.io",
                    "password_hash": hash_password("dev123"),
                    "name": "AI Engineer",
                    "role": "Lead Developer",
                    "created_at": time.time(),
                },
            ]
            self.save()

    def save(self):
        """Persists users to JSON storage."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.users, f, indent=2)

    def authenticate(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Validates email and password credentials. Returns sanitized user if valid."""
        email = email.strip().lower()
        target_hash = hash_password(password)
        for u in self.users:
            if u.get("email", "").lower() == email:
                if u.get("password_hash") == target_hash:
                    return self._sanitize(u)
                return None
        return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Finds a user by email address."""
        email = email.strip().lower()
        for u in self.users:
            if u.get("email", "").lower() == email:
                return self._sanitize(u)
        return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Finds a user by ID."""
        for u in self.users:
            if u.get("id") == user_id:
                return self._sanitize(u)
        return None

    def register_user(self, email: str, password: str, name: str = "", role: str = "Member") -> Dict[str, Any]:
        """Registers a new user account."""
        email = email.strip().lower()
        if not email or "@" not in email:
            raise ValueError("Valid email address is required")
        if not password or len(password) < 4:
            raise ValueError("Password must be at least 4 characters")

        for u in self.users:
            if u.get("email", "").lower() == email:
                raise ValueError("User with this email already exists")

        user_id = f"user-{int(time.time())}"
        new_user = {
            "id": user_id,
            "email": email,
            "password_hash": hash_password(password),
            "name": name.strip() or email.split("@")[0].capitalize(),
            "role": role,
            "created_at": time.time(),
        }
        self.users.append(new_user)
        self.save()
        return self._sanitize(new_user)

    def list_users(self) -> List[Dict[str, Any]]:
        """Returns list of sanitized users."""
        return [self._sanitize(u) for u in self.users]

    def generate_otp(self, email: str, purpose: str = "registration") -> str:
        """Generates a secure 6-digit numeric OTP and stores with 10-minute expiry."""
        email = email.strip().lower()
        otp_code = f"{random.randint(100000, 999999)}"
        key = f"{email}:{purpose}"
        self.otps[key] = {
            "code": otp_code,
            "expires_at": time.time() + 600,  # 10 minutes
            "purpose": purpose,
            "email": email,
        }
        return otp_code

    def verify_otp(self, email: str, otp: str, purpose: str = "registration") -> bool:
        """Verifies OTP code and invalidates it upon successful match."""
        email = email.strip().lower()
        otp = otp.strip()
        key = f"{email}:{purpose}"
        record = self.otps.get(key)
        if not record:
            return False
        
        # Check expiry
        if time.time() > record["expires_at"]:
            self.otps.pop(key, None)
            return False

        # Check code match
        if record["code"] == otp:
            self.otps.pop(key, None)
            return True
        return False

    def reset_password(self, email: str, new_password: str) -> bool:
        """Updates the password for a registered user."""
        email = email.strip().lower()
        if not new_password or len(new_password) < 4:
            raise ValueError("Password must be at least 4 characters")

        for u in self.users:
            if u.get("email", "").lower() == email:
                u["password_hash"] = hash_password(new_password)
                self.save()
                return True
        return False

    @staticmethod
    def _sanitize(user_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a safe representation of user omitting password_hash."""
        return {
            "id": user_dict.get("id"),
            "email": user_dict.get("email"),
            "name": user_dict.get("name"),
            "role": user_dict.get("role"),
            "avatar_url": user_dict.get("avatar_url", ""),
            "created_at": user_dict.get("created_at"),
            "updated_at": user_dict.get("updated_at"),
        }

    def issue_tokens(self, user_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates standard JWT access and refresh token pair for a user.
        """
        token_data = {
            "sub": str(user_dict.get("id", "")),
            "email": user_dict.get("email", ""),
            "name": user_dict.get("name", ""),
            "role": user_dict.get("role", ""),
            "avatar_url": user_dict.get("avatar_url", ""),
        }
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "token": access_token,  # Backward compatibility for legacy tests
        }

    def verify_token(self, token: str, expected_type: Optional[str] = "access") -> Dict[str, Any]:
        """Validates a JWT token and returns payload."""
        return decode_token(token, expected_type=expected_type)

    def refresh_access_token(self, refresh_token_str: str) -> Dict[str, Any]:
        """
        Validates refresh token and generates fresh access token & rotated refresh token.
        """
        payload = decode_token(refresh_token_str, expected_type="refresh")
        email = payload.get("email")
        if not email:
            raise ValueError("Refresh token missing user identity")
        user = self.get_user_by_email(email)
        if not user:
            raise ValueError("User account associated with this token is not found or inactive")
        return self.issue_tokens(user)

    def update_profile(
        self,
        user_id: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        current_password: Optional[str] = None,
        new_password: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Updates profile fields for a user: full name, email, password, and avatar photo.
        If the email address changes:
        - Validates email syntax and verifies uniqueness across all accounts.
        - Automatically transfers all organization ownerships, roles, memberships, and applications.
        """
        user = None
        for u in self.users:
            if u.get("id") == user_id:
                user = u
                break
        if not user:
            raise ValueError(f"User with ID '{user_id}' not found.")

        old_email = user.get("email", "").strip().lower()
        new_email = (email.strip().lower() if email is not None and email.strip() else old_email)

        # 1. Email change validation
        if new_email != old_email:
            if not new_email or "@" not in new_email:
                raise ValueError("A valid email address is required.")
            for other_user in self.users:
                if other_user.get("id") != user_id and other_user.get("email", "").strip().lower() == new_email:
                    raise ValueError(f"The email address '{new_email}' is already in use by another account.")

        # 2. Password change validation
        if new_password:
            if not current_password:
                raise ValueError("Current password is required to change your password.")
            if user.get("password_hash") != hash_password(current_password):
                raise ValueError("Current password does not match.")
            if len(new_password) < 4:
                raise ValueError("New password must be at least 4 characters.")
            user["password_hash"] = hash_password(new_password)

        # 3. Apply Name
        if name is not None:
            user["name"] = name.strip()

        # 4. Apply Avatar
        if avatar_url is not None:
            user["avatar_url"] = avatar_url.strip()

        # 5. Apply Email & transfer organization access
        if new_email != old_email:
            user["email"] = new_email
            try:
                from app.organizations import OrganizationManager
                om = OrganizationManager.get_instance()
                om.transfer_user_email(old_email, new_email, user.get("name"))
            except Exception as org_err:
                print(f"⚠️ [UserManager] Error transferring org access from {old_email} to {new_email}: {org_err}")

        user["updated_at"] = time.time()
        self.save()
        print(f"👤 [UserManager] Profile updated for user {user_id} (email: {user['email']}, name: {user.get('name')})")
        return self._sanitize(user)


