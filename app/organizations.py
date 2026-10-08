import json
import re
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

ORGANIZATIONS_FILE = Path(__file__).resolve().parent / "data" / "organizations.json"

VALID_ROLES = ["owner", "admin", "analyst", "user"]
APPLICABLE_ROLES = ["admin", "analyst", "user"]


def slugify(text: str) -> str:
    """Converts a title or name into a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-") or "custom-org"


class OrganizationManager:
    """
    Manages persistence, lifecycle, active state, roles (owner, admin, analyst, user),
    memberships, and entry applications for Organizations.
    """
    _instance: Optional["OrganizationManager"] = None

    def __init__(self, storage_path: Path = ORGANIZATIONS_FILE):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.organizations: List[Dict[str, Any]] = []
        self.load()

    @classmethod
    def get_instance(cls) -> "OrganizationManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Loads organizations from disk, or seeds default organization if absent."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.organizations = json.load(f)
            except Exception as e:
                print(f"⚠️ [OrganizationManager] Error reading {self.storage_path}: {e}")
                self.organizations = []

        if not self.organizations:
            self.organizations = [
                {
                    "id": "apex-ai-innovations",
                    "name": "Apex AI Innovations",
                    "description": "Enterprise conversational intelligence and voice automation hub managing specialized AI agents across telephone and web channels.",
                    "industry": "Technology & Software Services",
                    "owner_email": "admin@apexai.io",
                    "members": [
                        {
                            "email": "admin@apexai.io",
                            "name": "Platform Administrator",
                            "role": "owner",
                            "status": "approved",
                            "joined_at": 1789990000,
                        }
                    ],
                    "applications": [],
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "is_active": True,
                }
            ]
            self.save()
        else:
            # Migration/backward-compatibility: ensure owner_email, members and applications exist on all orgs
            modified = False
            for org in self.organizations:
                owner = org.get("owner_email") or "admin@apexai.io"
                if "owner_email" not in org:
                    org["owner_email"] = owner
                    modified = True

                if "members" not in org or not org["members"]:
                    org["members"] = [
                        {
                            "email": owner,
                            "name": "Platform Administrator" if owner == "admin@apexai.io" else owner.split("@")[0].title(),
                            "role": "owner",
                            "status": "approved",
                            "joined_at": org.get("created_at", time.time()),
                        }
                    ]
                    modified = True

                if "applications" not in org:
                    org["applications"] = []
                    modified = True

            if modified:
                self.save()

    def save(self) -> None:
        """Persists organizations registry to disk."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.organizations, f, indent=2)
        except Exception as e:
            print(f"⚠️ [OrganizationManager] Error saving to {self.storage_path}: {e}")

    def list_organizations(self, user_email: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns list of all registered organizations formatted with user-specific role details."""
        if not user_email:
            return self.organizations
        return [self.format_org_for_user(org, user_email) for org in self.organizations]

    def get_organization(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Finds an organization by id."""
        for org in self.organizations:
            if org.get("id") == org_id:
                return org
        return None

    def get_active_organization(self, user_email: Optional[str] = None) -> Dict[str, Any]:
        """Returns the currently active organization, or the first available."""
        target = None
        for org in self.organizations:
            if org.get("is_active"):
                target = org
                break
        if not target and self.organizations:
            self.organizations[0]["is_active"] = True
            self.save()
            target = self.organizations[0]
        elif not target:
            self.load()
            target = self.organizations[0]

        if user_email:
            return self.format_org_for_user(target, user_email)
        return target

    def set_active_organization(self, org_id: str, user_email: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Marks the specified organization as active, deactivating others."""
        target = self.get_organization(org_id)
        if not target:
            return None

        # Verify access permission if user_email provided
        if user_email and not self.can_user_access_space(org_id, user_email):
            raise PermissionError(f"Entry denied. User {user_email} is not an approved member of {target.get('name')}.")

        for org in self.organizations:
            if org.get("id") == org_id:
                org["is_active"] = True
            else:
                org["is_active"] = False

        self.save()
        if user_email:
            return self.format_org_for_user(target, user_email)
        return target

    def create_organization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a new organization with the creator assigned as owner."""
        name = data.get("name", "").strip() or "New Organization"
        description = data.get("description", "").strip() or "Specialized AI voice agent organization."
        industry = data.get("industry", "").strip() or "Technology"

        creator_email = (data.get("creator_email") or "admin@apexai.io").strip().lower()
        creator_name = (data.get("creator_name") or "").strip()
        if not creator_name:
            creator_name = "Platform Administrator" if creator_email == "admin@apexai.io" else creator_email.split("@")[0].title()

        base_id = slugify(name)
        org_id = base_id
        counter = 1
        existing_ids = {org.get("id") for org in self.organizations}
        while org_id in existing_ids:
            org_id = f"{base_id}-{counter}"
            counter += 1

        new_org = {
            "id": org_id,
            "name": name,
            "description": description,
            "industry": industry,
            "owner_email": creator_email,
            "members": [
                {
                    "email": creator_email,
                    "name": creator_name,
                    "role": "owner",
                    "status": "approved",
                    "joined_at": time.time(),
                }
            ],
            "applications": [],
            "created_at": time.time(),
            "updated_at": time.time(),
            "is_active": False,
        }

        self.organizations.append(new_org)
        self.save()
        return new_org

    def update_organization(self, org_id: str, data: Dict[str, Any], updater_email: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Updates an existing organization's details (restricted to owner)."""
        org = self.get_organization(org_id)
        if not org:
            return None

        if updater_email:
            role = self.get_user_role(org_id, updater_email)
            if role != "owner":
                raise PermissionError("Only the Organization Owner can edit organization details.")

        if "name" in data and data["name"].strip():
            org["name"] = data["name"].strip()
        if "description" in data:
            org["description"] = data["description"].strip()
        if "industry" in data and data["industry"].strip():
            org["industry"] = data["industry"].strip()

        org["updated_at"] = time.time()
        self.save()
        return org

    def delete_organization(self, org_id: str, deleter_email: Optional[str] = None) -> bool:
        """Deletes an organization by ID (minimum 1 required, restricted to owner)."""
        if len(self.organizations) <= 1:
            return False

        target = self.get_organization(org_id)
        if not target:
            return False

        if deleter_email:
            role = self.get_user_role(org_id, deleter_email)
            if role != "owner":
                raise PermissionError("Only the Organization Owner can delete this organization.")

        was_active = target.get("is_active", False)
        self.organizations = [o for o in self.organizations if o.get("id") != org_id]
        if was_active and self.organizations:
            self.organizations[0]["is_active"] = True

        self.save()
        return True

    # -------------------------------------------------------------
    # Roles & Membership Management
    # -------------------------------------------------------------
    def get_user_role(self, org_id: str, user_email: Optional[str]) -> Optional[str]:
        """Returns user's role in the organization ('owner', 'admin', 'analyst', 'user', or None)."""
        if not user_email:
            return None
        email_clean = user_email.strip().lower()
        org = self.get_organization(org_id)
        if not org:
            return None

        # Check explicit members list
        for m in org.get("members", []):
            if m.get("email", "").strip().lower() == email_clean and m.get("status") == "approved":
                return m.get("role", "user")

        # Fallback check for owner_email
        if org.get("owner_email", "").strip().lower() == email_clean:
            return "owner"

        return None

    def get_user_application(self, org_id: str, user_email: Optional[str]) -> Optional[Dict[str, Any]]:
        """Finds any entry application submitted by the user for this organization."""
        if not user_email:
            return None
        email_clean = user_email.strip().lower()
        org = self.get_organization(org_id)
        if not org:
            return None

        for app in org.get("applications", []):
            if app.get("applicant_email", "").strip().lower() == email_clean:
                return app
        return None

    def can_user_access_space(self, org_id: str, user_email: Optional[str]) -> bool:
        """Returns True if the user is an approved member of this organization."""
        return self.get_user_role(org_id, user_email) is not None

    def is_member(self, org_id: str, user_email: Optional[str]) -> bool:
        """Returns True if the user is a registered member (owner, admin, analyst, or user) of the organization."""
        return self.get_user_role(org_id, user_email) is not None

    def format_org_for_user(self, org: Dict[str, Any], user_email: Optional[str]) -> Dict[str, Any]:
        """Formats an organization object with computed role, application status, and permission flags for a given user."""
        res = dict(org)
        user_role = self.get_user_role(org.get("id", ""), user_email)
        user_app = self.get_user_application(org.get("id", ""), user_email)

        res["user_role"] = user_role
        res["user_application"] = user_app
        res["can_open"] = user_role is not None

        # Count pending applications for owners and admins
        if user_role in ["owner", "admin"]:
            pending = [a for a in org.get("applications", []) if a.get("status") == "pending"]
            res["pending_applications_count"] = len(pending)
        else:
            res["pending_applications_count"] = 0

        res["members_count"] = len([m for m in org.get("members", []) if m.get("status") == "approved"])
        return res

    # -------------------------------------------------------------
    # Application Flow (Apply for Entry & Review)
    # -------------------------------------------------------------
    def apply_for_entry(
        self,
        org_id: str,
        applicant_email: str,
        applicant_name: str,
        requested_role: str,
        note: str = "",
    ) -> Dict[str, Any]:
        """
        Submits an application for entrance into an organization.
        Applicant chooses role from: 'admin', 'analyst', 'user'.
        """
        org = self.get_organization(org_id)
        if not org:
            raise ValueError("Organization not found.")

        email_clean = applicant_email.strip().lower()
        name_clean = applicant_name.strip() or email_clean.split("@")[0].title()
        role_clean = requested_role.strip().lower()
        if role_clean == "developer":
            role_clean = "user"

        if role_clean not in APPLICABLE_ROLES:
            raise ValueError(f"Invalid role '{requested_role}'. Must be one of: {', '.join(APPLICABLE_ROLES)}")

        # Check if user is already an approved member
        current_role = self.get_user_role(org_id, email_clean)
        if current_role:
            raise ValueError(f"You are already an approved member of this organization with role '{current_role}'.")

        apps = org.setdefault("applications", [])
        existing = next((a for a in apps if a.get("applicant_email", "").strip().lower() == email_clean), None)

        if existing:
            existing["requested_role"] = role_clean
            existing["note"] = note.strip()
            existing["applicant_name"] = name_clean
            existing["status"] = "pending"
            existing["created_at"] = time.time()
            existing["reviewed_by"] = None
            existing["reviewed_at"] = None
            target_app = existing
        else:
            target_app = {
                "id": f"app-{int(time.time()*1000)}-{random.randint(100, 999)}",
                "org_id": org_id,
                "applicant_email": email_clean,
                "applicant_name": name_clean,
                "requested_role": role_clean,
                "note": note.strip(),
                "status": "pending",
                "created_at": time.time(),
                "reviewed_by": None,
                "reviewed_at": None,
            }
            apps.append(target_app)

        self.save()
        print(f"📝 [Organization Hub] New application submitted by {email_clean} for org '{org['name']}' (Requested: {role_clean})")
        return target_app

    def review_application(
        self,
        org_id: str,
        app_id: str,
        action: str,
        reviewer_email: str,
        approved_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Reviews an application. ONLY Owner and Admins can approve or deny entry.
        """
        org = self.get_organization(org_id)
        if not org:
            raise ValueError("Organization not found.")

        reviewer_role = self.get_user_role(org_id, reviewer_email)
        if reviewer_role not in ["owner", "admin"]:
            raise PermissionError("Only Organization Owners and Admins can approve or deny entry applications.")

        action_clean = action.strip().lower()
        if action_clean not in ["approve", "deny"]:
            raise ValueError("Action must be 'approve' or 'deny'.")

        apps = org.setdefault("applications", [])
        target_app = next((a for a in apps if a.get("id") == app_id), None)
        if not target_app:
            raise ValueError(f"Application with ID '{app_id}' not found.")

        target_app["reviewed_by"] = reviewer_email.strip().lower()
        target_app["reviewed_at"] = time.time()

        if action_clean == "approve":
            target_app["status"] = "approved"
            # Rule: Acceptance MUST be strictly as per the applicant's requested designation.
            # The reviewing Owner or Admin cannot assign a different designation upon approval;
            # if they disagree with the requested designation, they can deny the application.
            requested_role = (target_app.get("requested_role") or "user").strip().lower()
            if requested_role == "developer":
                requested_role = "user"
            if requested_role not in APPLICABLE_ROLES:
                requested_role = "user"

            # If reviewer attempted to pass an explicit role that contradicts the requested designation:
            if approved_role:
                clean_approved = approved_role.strip().lower()
                if clean_approved == "developer":
                    clean_approved = "user"
                if clean_approved != requested_role:
                    raise ValueError(
                        f"Acceptance must be strictly as per the requested designation '{requested_role}'. "
                        f"Owner or Admin cannot assign a different designation ('{clean_approved}'); "
                        "if you do not wish to grant this designation, you must deny the application."
                    )

            final_role = requested_role
            target_app["approved_role"] = final_role

            # Add to members list
            members = org.setdefault("members", [])
            applicant_email = target_app["applicant_email"]
            existing_m = next((m for m in members if m.get("email", "").strip().lower() == applicant_email), None)

            if existing_m:
                existing_m["role"] = final_role
                existing_m["status"] = "approved"
                existing_m["name"] = target_app.get("applicant_name", existing_m.get("name"))
            else:
                members.append({
                    "email": applicant_email,
                    "name": target_app.get("applicant_name", applicant_email.split("@")[0].title()),
                    "role": final_role,
                    "status": "approved",
                    "joined_at": time.time(),
                })
            print(f"✅ [Organization Hub] Approved {applicant_email} into '{org['name']}' as {final_role} (requested designation) by {reviewer_email}")
        else:
            target_app["status"] = "denied"
            # If was previously in members, remove
            org["members"] = [m for m in org.get("members", []) if m.get("email", "").strip().lower() != target_app["applicant_email"]]
            print(f"❌ [Organization Hub] Denied application {app_id} for {target_app['applicant_email']} by {reviewer_email}")

        self.save()
        return {
            "success": True,
            "application": target_app,
            "action": action_clean,
            "organization_id": org_id,
        }

    def list_applications(self, org_id: str, reviewer_email: str) -> List[Dict[str, Any]]:
        """Returns all applications for an organization (accessible only to owner/admin)."""
        role = self.get_user_role(org_id, reviewer_email)
        if role not in ["owner", "admin"]:
            raise PermissionError("Only Organization Owners and Admins can view entry applications.")
        org = self.get_organization(org_id)
        if not org:
            raise ValueError("Organization not found.")
        return org.get("applications", [])

    def list_members(self, org_id: str) -> List[Dict[str, Any]]:
        """Returns all approved members for an organization."""
        org = self.get_organization(org_id)
        if not org:
            raise ValueError("Organization not found.")
        return [m for m in org.get("members", []) if m.get("status") == "approved"]

    def add_member(
        self,
        org_id: str,
        email: str,
        name: str,
        role: str,
        adder_email: str,
    ) -> Dict[str, Any]:
        """Directly adds a member to the organization (owner or admin only)."""
        org = self.get_organization(org_id)
        if not org:
            raise ValueError("Organization not found.")

        adder_role = self.get_user_role(org_id, adder_email)
        if adder_role not in ["owner", "admin"]:
            raise PermissionError("Only Organization Owners and Admins can add members directly.")

        role_clean = role.strip().lower()
        if role_clean == "developer":
            role_clean = "user"
        if role_clean not in APPLICABLE_ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of: {', '.join(APPLICABLE_ROLES)}")

        email_clean = email.strip().lower()
        name_clean = name.strip() or email_clean.split("@")[0].title()

        members = org.setdefault("members", [])
        existing = next((m for m in members if m.get("email", "").strip().lower() == email_clean), None)

        if existing:
            if existing.get("role") == "owner":
                raise ValueError("Cannot override owner role.")
            existing["role"] = role_clean
            existing["status"] = "approved"
            existing["name"] = name_clean
            res_member = existing
        else:
            res_member = {
                "email": email_clean,
                "name": name_clean,
                "role": role_clean,
                "status": "approved",
                "joined_at": time.time(),
            }
            members.append(res_member)

        self.save()
        return res_member

    def invite_member(
        self,
        org_id: str,
        email: str,
        role: str,
        name: str = "",
        inviter_email: str = "",
    ) -> Dict[str, Any]:
        """
        Sends an invitation / designates a member with the designation chosen by the Owner or Admin.
        """
        return self.add_member(
            org_id=org_id,
            email=email,
            name=name,
            role=role,
            adder_email=inviter_email,
        )

    def update_member_role(
        self,
        org_id: str,
        email: str,
        new_role: str,
        updater_email: str,
    ) -> Dict[str, Any]:
        """Updates a member's role (owner or admin only; owner cannot be demoted)."""
        org = self.get_organization(org_id)
        if not org:
            raise ValueError("Organization not found.")

        updater_role = self.get_user_role(org_id, updater_email)
        if updater_role not in ["owner", "admin"]:
            raise PermissionError("Only Organization Owners and Admins can update member roles.")

        email_clean = email.strip().lower()
        role_clean = new_role.strip().lower()
        if role_clean == "developer":
            role_clean = "user"

        if role_clean not in APPLICABLE_ROLES:
            raise ValueError(f"Invalid role '{new_role}'. Must be one of: {', '.join(APPLICABLE_ROLES)}")

        members = org.setdefault("members", [])
        target = next((m for m in members if m.get("email", "").strip().lower() == email_clean), None)
        if not target:
            raise ValueError(f"Member '{email}' not found.")

        if target.get("role") == "owner":
            raise ValueError("Cannot change the role of the Organization Owner.")

        target["role"] = role_clean
        self.save()
        return target

    def remove_member(self, org_id: str, email: str, remover_email: str) -> bool:
        """Removes a member from the organization (owner or admin only; owner cannot be removed)."""
        org = self.get_organization(org_id)
        if not org:
            raise ValueError("Organization not found.")

        remover_role = self.get_user_role(org_id, remover_email)
        if remover_role not in ["owner", "admin"]:
            raise PermissionError("Only Organization Owners and Admins can remove members.")

        email_clean = email.strip().lower()
        members = org.setdefault("members", [])
        target = next((m for m in members if m.get("email", "").strip().lower() == email_clean), None)
        if not target:
            return False

        if target.get("role") == "owner":
            raise ValueError("Cannot remove the Organization Owner.")

        org["members"] = [m for m in members if m.get("email", "").strip().lower() != email_clean]
        self.save()
        return True

    def transfer_user_email(
        self, old_email: str, new_email: str, new_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transfers organization ownerships, memberships, and application records from
        an old email address to a new email address when a user updates their profile email.
        """
        old_clean = old_email.strip().lower()
        new_clean = new_email.strip().lower()
        if not old_clean or not new_clean or old_clean == new_clean:
            return {"transferred_orgs": [], "count": 0}

        transferred_orgs = []
        for org in self.organizations:
            modified = False

            # 1. Transfer organization ownership
            if org.get("owner_email", "").strip().lower() == old_clean:
                org["owner_email"] = new_clean
                modified = True

            # 2. Transfer member records
            for m in org.get("members", []):
                if m.get("email", "").strip().lower() == old_clean:
                    m["email"] = new_clean
                    if new_name:
                        m["name"] = new_name
                    modified = True

            # 3. Transfer application records
            for a in org.get("applications", []):
                if a.get("applicant_email", "").strip().lower() == old_clean:
                    a["applicant_email"] = new_clean
                    if new_name:
                        a["applicant_name"] = new_name
                    modified = True
                if a.get("reviewed_by", "") and a.get("reviewed_by", "").strip().lower() == old_clean:
                    a["reviewed_by"] = new_clean
                    modified = True

            if modified:
                org["updated_at"] = time.time()
                transferred_orgs.append(org.get("id"))

        if transferred_orgs:
            self.save()
            print(f"🔄 [OrganizationManager] Transferred access from {old_clean} to {new_clean} across {len(transferred_orgs)} organization(s): {transferred_orgs}")

        return {"transferred_orgs": transferred_orgs, "count": len(transferred_orgs)}

