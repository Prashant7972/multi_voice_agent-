import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import atomic_save_json

CONTACTS_FILE = Path(__file__).resolve().parent / "data" / "contacts.json"

COUNTRY_CODES = [
    {"code": "+1", "country": "United States / Canada", "iso": "US", "flag": "🇺🇸"},
    {"code": "+44", "country": "United Kingdom", "iso": "GB", "flag": "🇬🇧"},
    {"code": "+91", "country": "India", "iso": "IN", "flag": "🇮🇳"},
    {"code": "+61", "country": "Australia", "iso": "AU", "flag": "🇦🇺"},
    {"code": "+49", "country": "Germany", "iso": "DE", "flag": "🇩🇪"},
    {"code": "+33", "country": "France", "iso": "FR", "flag": "🇫🇷"},
    {"code": "+81", "country": "Japan", "iso": "JP", "flag": "🇯🇵"},
    {"code": "+65", "country": "Singapore", "iso": "SG", "flag": "🇸🇬"},
    {"code": "+971", "country": "United Arab Emirates", "iso": "AE", "flag": "🇦🇪"},
    {"code": "+41", "country": "Switzerland", "iso": "CH", "flag": "🇨🇭"},
    {"code": "+31", "country": "Netherlands", "iso": "NL", "flag": "🇳🇱"},
    {"code": "+34", "country": "Spain", "iso": "ES", "flag": "🇪🇸"},
    {"code": "+39", "country": "Italy", "iso": "IT", "flag": "🇮🇹"},
    {"code": "+55", "country": "Brazil", "iso": "BR", "flag": "🇧🇷"},
    {"code": "+52", "country": "Mexico", "iso": "MX", "flag": "🇲🇽"},
    {"code": "+27", "country": "South Africa", "iso": "ZA", "flag": "🇿🇦"},
    {"code": "+82", "country": "South Korea", "iso": "KR", "flag": "🇰🇷"},
    {"code": "+64", "country": "New Zealand", "iso": "NZ", "flag": "🇳🇿"},
    {"code": "+353", "country": "Ireland", "iso": "IE", "flag": "🇮🇪"},
    {"code": "+46", "country": "Sweden", "iso": "SE", "flag": "🇸🇪"},
]


def format_e164(country_code: str, raw_phone: str) -> str:
    """Normalizes country code and phone number into E.164 format (+[country_code][number])."""
    cc = country_code.strip()
    if not cc.startswith("+"):
        cc = f"+{cc}"
    # Strip any non-digit chars from raw phone
    digits = re.sub(r"\D", "", raw_phone)
    if digits.startswith("0"):
        digits = digits.lstrip("0")
    # If the user already typed country code digits at the start, don't duplicate
    cc_digits = cc.lstrip("+")
    if digits.startswith(cc_digits) and len(digits) > len(cc_digits) + 5:
        digits = digits[len(cc_digits):]
    return f"{cc}{digits}"


class ContactsManager:
    """
    Manages organization-scoped contacts and named contact groups.
    Persists data to app/data/contacts.json.
    """
    _instance: Optional["ContactsManager"] = None

    def __init__(self, storage_path: Path = CONTACTS_FILE):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.groups: List[Dict[str, Any]] = []
        self.contacts: List[Dict[str, Any]] = []
        self.load()

    @classmethod
    def get_instance(cls) -> "ContactsManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Loads contacts and groups from JSON, seeding defaults if absent."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.groups = data.get("groups", [])
                    self.contacts = data.get("contacts", [])
            except Exception as e:
                print(f"⚠️ [ContactsManager] Error reading {self.storage_path}: {e}")
                self.groups = []
                self.contacts = []

        if not self.groups and not self.contacts:
            self._seed_defaults()

    def _seed_defaults(self) -> None:
        """Seeds default named groups and sample contacts for apex-ai-innovations."""
        now = time.time()
        self.groups = [
            {
                "id": "grp-outreach-q4",
                "org_id": "apex-ai-innovations",
                "name": "Q4 Enterprise Outreach",
                "description": "High-priority enterprise executives for agent telephony demos.",
                "created_at": now - 86400,
                "created_by": "admin@apexai.io",
            },
            {
                "id": "grp-feedback-vip",
                "org_id": "apex-ai-innovations",
                "name": "VIP Customer Feedback",
                "description": "Key accounts participating in local AI voice quality evaluation.",
                "created_at": now - 43200,
                "created_by": "admin@apexai.io",
            },
            {
                "id": "grp-beta-testers",
                "org_id": "apex-ai-innovations",
                "name": "Beta Testers Group",
                "description": "Early access users testing latency and neural VAD performance.",
                "created_at": now - 20000,
                "created_by": "admin@apexai.io",
            },
        ]

        self.contacts = [
            {
                "id": "cnt-1001",
                "org_id": "apex-ai-innovations",
                "name": "Sarah Connor",
                "country_code": "+1",
                "phone": "4155550199",
                "full_phone": "+14155550199",
                "email": "sconnor@cyberdyne.io",
                "group_id": "grp-outreach-q4",
                "group_name": "Q4 Enterprise Outreach",
                "status": "active",
                "notes": "Interested in low-latency telephony agent",
                "created_at": now - 85000,
                "created_by": "admin@apexai.io",
            },
            {
                "id": "cnt-1002",
                "org_id": "apex-ai-innovations",
                "name": "Alexander Hayes",
                "country_code": "+44",
                "phone": "2079460912",
                "full_phone": "+442079460912",
                "email": "a.hayes@apexpartners.uk",
                "group_id": "grp-outreach-q4",
                "group_name": "Q4 Enterprise Outreach",
                "status": "active",
                "notes": "Requires UK compliance and GDPR verification",
                "created_at": now - 84000,
                "created_by": "admin@apexai.io",
            },
            {
                "id": "cnt-1003",
                "org_id": "apex-ai-innovations",
                "name": "Priya Sharma",
                "country_code": "+91",
                "phone": "9820155521",
                "full_phone": "+919820155521",
                "email": "priya.sharma@techflow.in",
                "group_id": "grp-feedback-vip",
                "group_name": "VIP Customer Feedback",
                "status": "active",
                "notes": "Testing multilingual accents and Piper voices",
                "created_at": now - 40000,
                "created_by": "admin@apexai.io",
            },
            {
                "id": "cnt-1004",
                "org_id": "apex-ai-innovations",
                "name": "Liam Vance",
                "country_code": "+61",
                "phone": "298765432",
                "full_phone": "+61298765432",
                "email": "liam.vance@sydneytech.au",
                "group_id": "grp-feedback-vip",
                "group_name": "VIP Customer Feedback",
                "status": "active",
                "notes": "High volume inbound inquiries",
                "created_at": now - 35000,
                "created_by": "admin@apexai.io",
            },
            {
                "id": "cnt-1005",
                "org_id": "apex-ai-innovations",
                "name": "Elena Rostova",
                "country_code": "+49",
                "phone": "3012345678",
                "full_phone": "+493012345678",
                "email": "elena.rostova@berlinconsult.de",
                "group_id": "grp-beta-testers",
                "group_name": "Beta Testers Group",
                "status": "active",
                "notes": "Evaluated neural VAD and barge-in",
                "created_at": now - 18000,
                "created_by": "admin@apexai.io",
            },
        ]
        self.save()

    def save(self) -> None:
        """Persists groups and contacts to JSON atomically."""
        try:
            atomic_save_json(self.storage_path, {
                "groups": self.groups,
                "contacts": self.contacts,
            })
        except Exception as e:
            print(f"⚠️ [ContactsManager] Error saving to {self.storage_path}: {e}")

    # --- Groups CRUD ---

    def list_groups(self, org_id: str) -> List[Dict[str, Any]]:
        """Returns all named contact groups for an organization with contact counts."""
        org_id_clean = org_id.strip().lower()
        res = []
        for g in self.groups:
            if g.get("org_id", "").strip().lower() == org_id_clean:
                # Count contacts in this group
                count = sum(1 for c in self.contacts if c.get("org_id", "").strip().lower() == org_id_clean and c.get("group_id") == g.get("id"))
                group_data = dict(g)
                group_data["contact_count"] = count
                res.append(group_data)
        return res

    def get_group(self, org_id: str, group_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a specific group by ID."""
        org_id_clean = org_id.strip().lower()
        for g in self.groups:
            if g.get("org_id", "").strip().lower() == org_id_clean and g.get("id") == group_id:
                count = sum(1 for c in self.contacts if c.get("org_id", "").strip().lower() == org_id_clean and c.get("group_id") == group_id)
                group_data = dict(g)
                group_data["contact_count"] = count
                return group_data
        return None

    def create_group(self, org_id: str, name: str, description: str = "", created_by: str = "") -> Dict[str, Any]:
        """Creates a new named contact group for the organization."""
        if not name or not name.strip():
            raise ValueError("Group name cannot be empty.")
        org_id_clean = org_id.strip().lower()
        group_id = f"grp-{int(time.time() * 1000)}"
        new_group = {
            "id": group_id,
            "org_id": org_id_clean,
            "name": name.strip(),
            "description": description.strip(),
            "created_at": time.time(),
            "created_by": created_by.strip() or "system",
            "contact_count": 0,
        }
        self.groups.append(new_group)
        self.save()
        return new_group

    def update_group(self, org_id: str, group_id: str, name: Optional[str] = None, description: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Updates group name or description."""
        org_id_clean = org_id.strip().lower()
        for g in self.groups:
            if g.get("org_id", "").strip().lower() == org_id_clean and g.get("id") == group_id:
                if name is not None and name.strip():
                    g["name"] = name.strip()
                    # Also update group_name cached on contacts
                    for c in self.contacts:
                        if c.get("group_id") == group_id:
                            c["group_name"] = name.strip()
                if description is not None:
                    g["description"] = description.strip()
                self.save()
                return g
        return None

    def delete_group(self, org_id: str, group_id: str) -> bool:
        """Deletes a group and unassigns or deletes contacts."""
        org_id_clean = org_id.strip().lower()
        initial_len = len(self.groups)
        self.groups = [g for g in self.groups if not (g.get("org_id", "").strip().lower() == org_id_clean and g.get("id") == group_id)]
        if len(self.groups) < initial_len:
            # Reassign contacts in this group to unassigned
            for c in self.contacts:
                if c.get("org_id", "").strip().lower() == org_id_clean and c.get("group_id") == group_id:
                    c["group_id"] = ""
                    c["group_name"] = "Unassigned"
            self.save()
            return True
        return False

    # --- Contacts CRUD ---

    def list_contacts(
        self,
        org_id: str,
        group_id: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Returns contacts scoped to an org, optionally filtered by group or search keyword."""
        org_id_clean = org_id.strip().lower()
        res = [c for c in self.contacts if c.get("org_id", "").strip().lower() == org_id_clean]

        if group_id:
            res = [c for c in res if c.get("group_id") == group_id]

        if search and search.strip():
            q = search.strip().lower()
            res = [
                c for c in res
                if q in c.get("name", "").lower()
                or q in c.get("phone", "").lower()
                or q in c.get("full_phone", "").lower()
                or q in c.get("email", "").lower()
                or q in c.get("group_name", "").lower()
            ]

        # Sort newest first
        res.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return res

    def get_contact(self, org_id: str, contact_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a single contact by ID."""
        org_id_clean = org_id.strip().lower()
        for c in self.contacts:
            if c.get("org_id", "").strip().lower() == org_id_clean and c.get("id") == contact_id:
                return c
        return None

    def create_contact(
        self,
        org_id: str,
        name: str,
        country_code: str,
        phone: str,
        group_id: Optional[str] = None,
        email: Optional[str] = None,
        notes: Optional[str] = None,
        created_by: str = "",
    ) -> Dict[str, Any]:
        """Creates a new contact with country code and phone number."""
        if not name or not name.strip():
            raise ValueError("Contact name is required.")
        if not phone or not phone.strip():
            raise ValueError("Phone number is required.")
        if not country_code or not country_code.strip():
            country_code = "+1"

        org_id_clean = org_id.strip().lower()
        full_phone = format_e164(country_code, phone)

        # Lookup group name if group_id is provided
        group_name = "Unassigned"
        if group_id:
            grp = self.get_group(org_id_clean, group_id)
            if grp:
                group_name = grp.get("name", "Unassigned")
            else:
                group_id = ""

        contact_id = f"cnt-{int(time.time() * 1000)}"
        new_contact = {
            "id": contact_id,
            "org_id": org_id_clean,
            "name": name.strip(),
            "country_code": country_code.strip(),
            "phone": re.sub(r"\D", "", phone),
            "full_phone": full_phone,
            "email": (email or "").strip(),
            "group_id": group_id or "",
            "group_name": group_name,
            "status": "active",
            "notes": (notes or "").strip(),
            "created_at": time.time(),
            "created_by": created_by.strip() or "system",
        }

        self.contacts.append(new_contact)
        self.save()
        return new_contact

    def update_contact(
        self,
        org_id: str,
        contact_id: str,
        name: Optional[str] = None,
        country_code: Optional[str] = None,
        phone: Optional[str] = None,
        group_id: Optional[str] = None,
        email: Optional[str] = None,
        status: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Updates contact details."""
        org_id_clean = org_id.strip().lower()
        for c in self.contacts:
            if c.get("org_id", "").strip().lower() == org_id_clean and c.get("id") == contact_id:
                if name is not None and name.strip():
                    c["name"] = name.strip()
                if country_code is not None and country_code.strip():
                    c["country_code"] = country_code.strip()
                if phone is not None and phone.strip():
                    c["phone"] = re.sub(r"\D", "", phone)
                # Recalculate full_phone if either changed
                c["full_phone"] = format_e164(c["country_code"], c["phone"])
                if group_id is not None:
                    c["group_id"] = group_id
                    if group_id:
                        grp = self.get_group(org_id_clean, group_id)
                        c["group_name"] = grp.get("name", "Unassigned") if grp else "Unassigned"
                    else:
                        c["group_name"] = "Unassigned"
                if email is not None:
                    c["email"] = email.strip()
                if status is not None and status.strip():
                    c["status"] = status.strip()
                if notes is not None:
                    c["notes"] = notes.strip()
                self.save()
                return c
        return None

    def delete_contact(self, org_id: str, contact_id: str) -> bool:
        """Deletes a contact."""
        org_id_clean = org_id.strip().lower()
        initial_len = len(self.contacts)
        self.contacts = [c for c in self.contacts if not (c.get("org_id", "").strip().lower() == org_id_clean and c.get("id") == contact_id)]
        if len(self.contacts) < initial_len:
            self.save()
            return True
        return False
