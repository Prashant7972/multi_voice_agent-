import os
import json
import time
import uuid
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.organizations import OrganizationManager

DATA_DIR = Path(__file__).resolve().parent / "data"
MESSAGING_FILE = DATA_DIR / "messaging.json"
CHAT_MEDIA_DIR = DATA_DIR / "chat_media"


def detect_file_type(filename: str, content_type: Optional[str] = None) -> str:
    """Classifies file as 'photo', 'video', or 'document'."""
    lower_fn = filename.lower()
    ct = (content_type or "").lower()

    if ct.startswith("image/") or lower_fn.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp")):
        return "photo"
    if ct.startswith("video/") or lower_fn.endswith((".mp4", ".webm", ".mov", ".m4v", ".ogg", ".mkv")):
        return "video"
    return "document"


class MessagingManager:
    """
    Manages real-time conversations (direct & group), messages, media attachments,
    reactions, and group admin permissions scoped strictly per organization.
    """
    _instance: Optional["MessagingManager"] = None

    def __init__(self, storage_path: Path = MESSAGING_FILE, media_dir: Path = CHAT_MEDIA_DIR):
        self.storage_path = storage_path
        self.media_dir = media_dir
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)

        self.conversations: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []
        self.load()

    @classmethod
    def get_instance(cls) -> "MessagingManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Loads messaging state from disk or seeds initial channels."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.conversations = data.get("conversations", [])
                    self.messages = data.get("messages", [])
                    return
            except Exception as e:
                print(f"⚠️ [MessagingManager] Failed to read {self.storage_path}: {e}")
                self.conversations = []
                self.messages = []

        # Seed default public channel for default organization
        default_org_id = "apex-ai-innovations"
        default_admin = "admin@apexai.io"
        now = time.time()
        initial_conv_id = f"conv-group-{default_org_id}-general"

        self.conversations = [
            {
                "id": initial_conv_id,
                "org_id": default_org_id,
                "type": "group",
                "name": "General Team Hub",
                "description": "General organization channel for announcements and collaboration.",
                "creator_email": default_admin,
                "admins": [default_admin],
                "members": [default_admin],
                "created_at": now - 3600,
                "updated_at": now - 3600,
                "last_message": {
                    "text": "Welcome to the Apex Voice AI team messaging space!",
                    "sender_name": "Platform Administrator",
                    "timestamp": now - 3600,
                },
            }
        ]
        self.messages = [
            {
                "id": f"msg-{int(now * 1000)}-welcome",
                "conversation_id": initial_conv_id,
                "org_id": default_org_id,
                "sender_email": default_admin,
                "sender_name": "Platform Administrator",
                "text": "Welcome to the Apex Voice AI team messaging space! Share documents, images, videos, and collaborate with your team in real time.",
                "attachments": [],
                "reactions": {},
                "created_at": now - 3600,
            }
        ]
        self.save()

    def save(self) -> None:
        """Atomically saves conversations and messages to disk."""
        try:
            temp_path = self.storage_path.with_suffix(".tmp")
            data = {
                "conversations": self.conversations,
                "messages": self.messages,
            }
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.storage_path)
        except Exception as e:
            print(f"⚠️ [MessagingManager] Error writing to {self.storage_path}: {e}")

    # -------------------------------------------------------------------------
    # Conversation & Group Management
    # -------------------------------------------------------------------------

    def list_conversations(self, org_id: str, user_email: str) -> List[Dict[str, Any]]:
        """Returns all group channels and direct chats in org_id where user_email is a member."""
        org_manager = OrganizationManager.get_instance()
        org_members = {m["email"]: m.get("name", m["email"].split("@")[0].title()) 
                       for m in org_manager.list_members(org_id)}

        results = []
        for c in self.conversations:
            if c.get("org_id") != org_id:
                continue
            members = c.get("members", [])
            if user_email not in members:
                continue

            conv_copy = dict(c)
            # Decorate direct chat with counterparty name/email
            if conv_copy.get("type") == "direct":
                other_emails = [e for e in members if e != user_email]
                target_email = other_emails[0] if other_emails else user_email
                target_name = org_members.get(target_email, target_email.split("@")[0].title())
                conv_copy["display_name"] = target_name
                conv_copy["target_email"] = target_email
            else:
                conv_copy["display_name"] = conv_copy.get("name", "Unnamed Group")

            conv_copy["is_admin"] = user_email in conv_copy.get("admins", [])
            results.append(conv_copy)

        # Sort by updated_at descending
        results.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return results

    def get_conversation(self, org_id: str, conversation_id: str, user_email: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Gets a conversation by id within an organization."""
        for c in self.conversations:
            if c.get("id") == conversation_id and c.get("org_id") == org_id:
                if user_email and user_email not in c.get("members", []):
                    return None
                return dict(c)
        return None

    def create_group(
        self,
        org_id: str,
        creator_email: str,
        name: str,
        description: str = "",
        member_emails: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Creates an organization-scoped group channel.
        The creator is automatically granted the Group Admin role.
        Only approved members of org_id can be added.
        """
        org_manager = OrganizationManager.get_instance()
        approved_emails = {m["email"] for m in org_manager.list_members(org_id)}

        if creator_email not in approved_emails:
            raise ValueError("Creator must be an approved member of this organization.")

        name = name.strip()
        if not name:
            raise ValueError("Group name is required.")

        # Ensure unique, valid members within this org
        members_set = {creator_email}
        if member_emails:
            for em in member_emails:
                cleaned = em.strip().lower()
                if cleaned in approved_emails:
                    members_set.add(cleaned)

        now = time.time()
        conv_id = f"conv-group-{org_id}-{int(now * 1000)}-{uuid.uuid4().hex[:6]}"

        new_group = {
            "id": conv_id,
            "org_id": org_id,
            "type": "group",
            "name": name,
            "description": description.strip(),
            "creator_email": creator_email,
            "admins": [creator_email],
            "members": list(members_set),
            "created_at": now,
            "updated_at": now,
            "last_message": {
                "text": f"Group created by {creator_email.split('@')[0].title()}",
                "sender_name": "System",
                "timestamp": now,
            },
        }

        self.conversations.append(new_group)

        # Add initial system welcome message
        self.save_message(
            org_id=org_id,
            conversation_id=conv_id,
            sender_email="system@apexai.io",
            sender_name="System",
            text=f"{creator_email.split('@')[0].title()} created the group '{name}'.",
        )

        self.save()
        return new_group

    def get_or_create_direct_chat(
        self,
        org_id: str,
        user1_email: str,
        user2_email: str,
    ) -> Dict[str, Any]:
        """
        Gets existing 1-on-1 direct conversation or creates a new one between 2 org members.
        """
        org_manager = OrganizationManager.get_instance()
        approved_emails = {m["email"] for m in org_manager.list_members(org_id)}

        if user1_email not in approved_emails or user2_email not in approved_emails:
            raise ValueError("Both participants must be approved members of this organization.")

        pair = sorted([user1_email.lower(), user2_email.lower()])
        # Check existing direct chat
        for c in self.conversations:
            if c.get("org_id") == org_id and c.get("type") == "direct":
                c_members = sorted([m.lower() for m in c.get("members", [])])
                if c_members == pair:
                    return c

        # Create new direct chat
        now = time.time()
        conv_id = f"conv-dm-{org_id}-{int(now * 1000)}-{uuid.uuid4().hex[:6]}"
        new_dm = {
            "id": conv_id,
            "org_id": org_id,
            "type": "direct",
            "name": f"Direct: {user1_email} & {user2_email}",
            "description": "1-on-1 Direct Message",
            "creator_email": user1_email,
            "admins": pair,
            "members": pair,
            "created_at": now,
            "updated_at": now,
            "last_message": None,
        }
        self.conversations.append(new_dm)
        self.save()
        return new_dm

    def add_group_member(
        self,
        org_id: str,
        conversation_id: str,
        actor_email: str,
        new_member_email: str,
    ) -> Dict[str, Any]:
        """
        Adds an approved organization member to a group channel.
        Permission: ONLY Group Admins can add members.
        """
        conv = None
        for c in self.conversations:
            if c.get("id") == conversation_id and c.get("org_id") == org_id:
                conv = c
                break

        if not conv:
            raise ValueError("Group conversation not found.")
        if conv.get("type") != "group":
            raise ValueError("Cannot add members to a direct message chat.")

        admins = conv.get("admins", [])
        if actor_email not in admins:
            raise PermissionError("Only Group Admins have permission to add members to this group.")

        org_manager = OrganizationManager.get_instance()
        approved_emails = {m["email"] for m in org_manager.list_members(org_id)}
        if new_member_email not in approved_emails:
            raise ValueError("User must be an approved member of this organization.")

        members = conv.get("members", [])
        if new_member_email in members:
            return conv  # already in group

        members.append(new_member_email)
        conv["members"] = members
        conv["updated_at"] = time.time()

        # Post system message
        self.save_message(
            org_id=org_id,
            conversation_id=conversation_id,
            sender_email="system@apexai.io",
            sender_name="System",
            text=f"{actor_email.split('@')[0].title()} added {new_member_email.split('@')[0].title()} to the group.",
        )

        self.save()
        return conv

    def remove_group_member(
        self,
        org_id: str,
        conversation_id: str,
        actor_email: str,
        target_email: str,
    ) -> Dict[str, Any]:
        """
        Removes a member from a group.
        Permission: ONLY Group Admins can remove other members (or member leaving on their own).
        """
        conv = None
        for c in self.conversations:
            if c.get("id") == conversation_id and c.get("org_id") == org_id:
                conv = c
                break

        if not conv:
            raise ValueError("Group conversation not found.")
        if conv.get("type") != "group":
            raise ValueError("Cannot remove members from a direct chat.")

        admins = conv.get("admins", [])
        is_self_leaving = (actor_email == target_email)
        if not is_self_leaving and actor_email not in admins:
            raise PermissionError("Only Group Admins have permission to remove members from this group.")

        members = conv.get("members", [])
        if target_email not in members:
            return conv

        # Prevent removing the only admin if members remain
        if target_email in admins and len(admins) == 1 and len(members) > 1 and is_self_leaving:
            # Promote next member to admin before leaving
            remaining = [m for m in members if m != target_email]
            if remaining:
                admins.append(remaining[0])

        members = [m for m in members if m != target_email]
        conv["members"] = members
        conv["admins"] = [a for a in admins if a != target_email]
        conv["updated_at"] = time.time()

        act_name = actor_email.split("@")[0].title()
        tar_name = target_email.split("@")[0].title()
        system_text = f"{tar_name} left the group." if is_self_leaving else f"{act_name} removed {tar_name} from the group."

        self.save_message(
            org_id=org_id,
            conversation_id=conversation_id,
            sender_email="system@apexai.io",
            sender_name="System",
            text=system_text,
        )

        self.save()
        return conv

    def promote_group_admin(
        self,
        org_id: str,
        conversation_id: str,
        actor_email: str,
        target_email: str,
    ) -> Dict[str, Any]:
        """
        Promotes an existing group member to Group Admin.
        Permission: ONLY Group Admins can promote other members to Admin.
        """
        conv = None
        for c in self.conversations:
            if c.get("id") == conversation_id and c.get("org_id") == org_id:
                conv = c
                break

        if not conv:
            raise ValueError("Group conversation not found.")
        if conv.get("type") != "group":
            raise ValueError("Cannot assign admins in a direct chat.")

        admins = conv.get("admins", [])
        if actor_email not in admins:
            raise PermissionError("Only Group Admins have permission to promote members to Admin.")

        members = conv.get("members", [])
        if target_email not in members:
            raise ValueError("Target user must be a member of this group before becoming an Admin.")

        if target_email not in admins:
            admins.append(target_email)
            conv["admins"] = admins
            conv["updated_at"] = time.time()

            self.save_message(
                org_id=org_id,
                conversation_id=conversation_id,
                sender_email="system@apexai.io",
                sender_name="System",
                text=f"{actor_email.split('@')[0].title()} promoted {target_email.split('@')[0].title()} to Group Admin.",
            )
            self.save()

        return conv

    # -------------------------------------------------------------------------
    # Messages & Attachments
    # -------------------------------------------------------------------------

    def get_messages(
        self,
        org_id: str,
        conversation_id: str,
        user_email: str,
        limit: int = 100,
        before_timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Returns message history for a conversation if user is a member."""
        conv = self.get_conversation(org_id, conversation_id, user_email)
        if not conv:
            raise PermissionError("Access denied. You are not a member of this conversation.")

        msgs = [m for m in self.messages if m.get("conversation_id") == conversation_id and m.get("org_id") == org_id]
        if before_timestamp:
            msgs = [m for m in msgs if m.get("created_at", 0) < before_timestamp]

        msgs.sort(key=lambda m: m.get("created_at", 0))
        return msgs[-limit:]

    def save_message(
        self,
        org_id: str,
        conversation_id: str,
        sender_email: str,
        sender_name: str,
        text: str = "",
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Saves a new message and updates conversation last_message preview."""
        now = time.time()
        msg_id = f"msg-{int(now * 1000)}-{uuid.uuid4().hex[:6]}"

        new_msg = {
            "id": msg_id,
            "conversation_id": conversation_id,
            "org_id": org_id,
            "sender_email": sender_email,
            "sender_name": sender_name or sender_email.split("@")[0].title(),
            "text": text.strip() if text else "",
            "attachments": attachments or [],
            "reactions": {},
            "created_at": now,
        }

        self.messages.append(new_msg)

        # Update conversation metadata
        for c in self.conversations:
            if c.get("id") == conversation_id and c.get("org_id") == org_id:
                snippet = text.strip() if text else ("Shared attachment" if attachments else "")
                c["last_message"] = {
                    "text": snippet[:80],
                    "sender_name": new_msg["sender_name"],
                    "timestamp": now,
                }
                c["updated_at"] = now
                break

        self.save()
        return new_msg

    def toggle_reaction(
        self,
        org_id: str,
        conversation_id: str,
        message_id: str,
        user_email: str,
        emoji: str,
    ) -> Dict[str, Any]:
        """Adds or removes an emoji reaction for user_email on message_id."""
        msg = None
        for m in self.messages:
            if m.get("id") == message_id and m.get("conversation_id") == conversation_id and m.get("org_id") == org_id:
                msg = m
                break

        if not msg:
            raise ValueError("Message not found.")

        reactions = msg.setdefault("reactions", {})
        user_list = reactions.setdefault(emoji, [])

        if user_email in user_list:
            user_list.remove(user_email)
            if not user_list:
                del reactions[emoji]
        else:
            user_list.append(user_email)

        self.save()
        return {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "reactions": reactions,
        }

    def save_attachment(
        self,
        org_id: str,
        filename: str,
        file_bytes: bytes,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Saves uploaded media (document, photo, video) to storage and returns metadata."""
        file_type = detect_file_type(filename, content_type)
        ext = Path(filename).suffix
        safe_name = f"media-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}{ext}"

        org_media_dir = self.media_dir / org_id
        org_media_dir.mkdir(parents=True, exist_ok=True)
        dest_path = org_media_dir / safe_name

        with open(dest_path, "wb") as f:
            f.write(file_bytes)

        url = f"/api/organizations/{org_id}/messages/media/{safe_name}"
        return {
            "id": f"att-{uuid.uuid4().hex[:8]}",
            "filename": filename,
            "saved_name": safe_name,
            "file_type": file_type,
            "mime_type": content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
            "size": len(file_bytes),
            "url": url,
        }

    def clear_conversation(
        self,
        org_id: str,
        conversation_id: str,
        actor_email: str,
    ) -> Dict[str, Any]:
        """
        Clears all messages in a conversation.
        Permission: ONLY Organization Owner, Organization Admin, or Group Admin.
        """
        conv = self.get_conversation(org_id, conversation_id)
        if not conv:
            raise ValueError("Conversation not found.")

        org_manager = OrganizationManager.get_instance()
        user_role = org_manager.get_user_role(org_id, actor_email)
        is_org_owner_or_admin = user_role in ["owner", "admin"]
        is_group_admin = actor_email in conv.get("admins", [])

        if not (is_org_owner_or_admin or is_group_admin):
            raise PermissionError("Only Organization Owners, Admins, or Group Admins can delete chat history.")

        # Remove all messages for this conversation
        self.messages = [
            m for m in self.messages
            if not (m.get("conversation_id") == conversation_id and m.get("org_id") == org_id)
        ]

        # Reset last_message preview
        for c in self.conversations:
            if c.get("id") == conversation_id and c.get("org_id") == org_id:
                c["last_message"] = None
                c["updated_at"] = time.time()
                break

        # Record system notice that chat was cleared
        actor_name = actor_email.split("@")[0].title()
        self.save_message(
            org_id=org_id,
            conversation_id=conversation_id,
            sender_email="system@apexai.io",
            sender_name="System",
            text=f"Chat history was cleared by {actor_name}.",
        )

        self.save()
        return {"conversation_id": conversation_id, "cleared_by": actor_email}

    def delete_message(
        self,
        org_id: str,
        conversation_id: str,
        message_id: str,
        actor_email: str,
    ) -> Dict[str, Any]:
        """
        Deletes a single message.
        Permission: ONLY Organization Owner, Admin, Group Admin, or original sender.
        """
        conv = self.get_conversation(org_id, conversation_id)
        if not conv:
            raise ValueError("Conversation not found.")

        msg = None
        for m in self.messages:
            if m.get("id") == message_id and m.get("conversation_id") == conversation_id and m.get("org_id") == org_id:
                msg = m
                break

        if not msg:
            raise ValueError("Message not found.")

        org_manager = OrganizationManager.get_instance()
        user_role = org_manager.get_user_role(org_id, actor_email)
        is_org_owner_or_admin = user_role in ["owner", "admin"]
        is_group_admin = actor_email in conv.get("admins", [])
        is_sender = (msg.get("sender_email") == actor_email)

        if not (is_org_owner_or_admin or is_group_admin or is_sender):
            raise PermissionError("Only Organization Owners, Admins, or the original sender can delete this message.")

        self.messages = [m for m in self.messages if m.get("id") != message_id]

        # Update last_message if the deleted message was the last one
        remaining = [m for m in self.messages if m.get("conversation_id") == conversation_id and m.get("org_id") == org_id]
        last_m = remaining[-1] if remaining else None
        for c in self.conversations:
            if c.get("id") == conversation_id and c.get("org_id") == org_id:
                if last_m:
                    snippet = last_m.get("text", "") or "Shared attachment"
                    c["last_message"] = {
                        "text": snippet[:80],
                        "sender_name": last_m.get("sender_name", "User"),
                        "timestamp": last_m.get("created_at", time.time()),
                    }
                else:
                    c["last_message"] = None
                c["updated_at"] = time.time()
                break

        self.save()
        return {"conversation_id": conversation_id, "message_id": message_id}

