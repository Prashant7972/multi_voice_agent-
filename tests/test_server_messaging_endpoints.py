import sys
import unittest
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.server import app, org_manager, messaging_manager

client = TestClient(app)

# Authenticate test client with Admin JWT token
_login_res = client.post("/api/auth/login", json={"email": "admin@apexai.io", "password": "admin123"})
_admin_token = _login_res.json()["access_token"]
client.headers["Authorization"] = f"Bearer {_admin_token}"


class TestServerMessagingEndpoints(unittest.TestCase):
    def setUp(self):
        self.org_id = "apex-ai-innovations"
        self.admin_email = "admin@apexai.io"

        # Ensure dev@apexai.io is in org
        approved_members = [m["email"] for m in org_manager.list_members(self.org_id)]
        if "dev@apexai.io" not in approved_members:
            org_manager.add_member(
                org_id=self.org_id,
                email="dev@apexai.io",
                name="AI Engineer",
                role="admin",
                adder_email=self.admin_email
            )
        if "qa@apexai.io" not in approved_members:
            org_manager.add_member(
                org_id=self.org_id,
                email="qa@apexai.io",
                name="QA Lead",
                role="user",
                adder_email=self.admin_email
            )

    def test_messaging_ui_elements(self):
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.text

        # 1. Sidebar hierarchy: Messaging directly below Knowledge Base
        self.assertIn('id="sidebarNavKnowledge"', html)
        self.assertIn('id="sidebarNavMessaging"', html)
        self.assertTrue(
            html.find('id="sidebarNavKnowledge"') < html.find('id="sidebarNavMessaging"'),
            "Messaging button must be placed directly below Knowledge Base in the sidebar."
        )

        # 2. Workspace subnav contains Messaging between Knowledge and Settings
        self.assertIn('id="btnSubnavKnowledge"', html)
        self.assertIn('id="btnSubnavMessaging"', html)
        self.assertIn('id="btnSubnavSettings"', html)
        self.assertTrue(
            html.find('id="btnSubnavKnowledge"') < html.find('id="btnSubnavMessaging"') < html.find('id="btnSubnavSettings"'),
            "Subnav should place Messaging between Knowledge Base and Settings."
        )

        # 3. Messaging Dashboard with the two primary actions: New Chat & Create Group
        self.assertIn('id="orgSectionMessaging"', html)
        self.assertIn('id="btnNewDirectChat"', html)
        self.assertIn('id="btnCreateGroup"', html)
        self.assertIn('New Chat', html)
        self.assertIn('Create Group', html)

        # 4. Filter tabs and search
        self.assertIn('id="tabMsgFilterAll"', html)
        self.assertIn('id="tabMsgFilterGroups"', html)
        self.assertIn('id="tabMsgFilterDirect"', html)
        self.assertIn('id="messagingSearchInput"', html)

        # 5. Active chat components
        self.assertIn('id="messagingChatPane"', html)
        self.assertIn('id="messagingMessagesContainer"', html)
        self.assertIn('id="chatMessageInput"', html)
        self.assertIn('id="chatMediaFileInput"', html)
        self.assertIn('id="chatEmojiPicker"', html)
        self.assertIn('id="messagingTypingIndicator"', html)
        self.assertIn('id="stagedAttachmentPreview"', html)

        # 6. Modals for New Chat, Create Group, Group Management, Lightbox
        self.assertIn('id="createGroupModal"', html)
        self.assertIn('id="createGroupMembersList"', html)
        self.assertIn('id="newChatModal"', html)
        self.assertIn('id="newChatMembersList"', html)
        self.assertIn('id="groupManageModal"', html)
        self.assertIn('id="groupManageAddSection"', html)
        self.assertIn('id="groupManageMembersList"', html)
        self.assertIn('id="mediaLightboxModal"', html)

    def test_available_members(self):
        res = client.get(f"/api/organizations/{self.org_id}/members/available")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        emails = [m["email"] for m in data.get("members", [])]
        self.assertIn(self.admin_email, emails)
        self.assertIn("dev@apexai.io", emails)

    def test_group_creation_and_membership_flow(self):
        # 1. Create a group
        create_res = client.post(
            f"/api/organizations/{self.org_id}/messages/groups",
            json={
                "name": "Backend Architecture Hub",
                "description": "API designs & performance optimization",
                "creator_email": self.admin_email,
                "member_emails": ["dev@apexai.io"],
            }
        )
        self.assertEqual(create_res.status_code, 200)
        group = create_res.json()["group"]
        conv_id = group["id"]
        self.assertEqual(group["name"], "Backend Architecture Hub")
        self.assertIn(self.admin_email, group["admins"])

        # 2. List conversations
        list_res = client.get(f"/api/organizations/{self.org_id}/messages/conversations?user_email={self.admin_email}")
        self.assertEqual(list_res.status_code, 200)
        convs = list_res.json()["conversations"]
        self.assertTrue(any(c["id"] == conv_id for c in convs))

        # 3. Admin adds qa@apexai.io
        add_res = client.post(
            f"/api/organizations/{self.org_id}/messages/conversations/{conv_id}/members/add",
            json={
                "actor_email": self.admin_email,
                "new_member_email": "qa@apexai.io",
            }
        )
        self.assertEqual(add_res.status_code, 200)
        updated_conv = add_res.json()["conversation"]
        self.assertIn("qa@apexai.io", updated_conv["members"])

        # 4. Non-admin (qa@apexai.io) cannot promote to admin
        unauth_promote = client.post(
            f"/api/organizations/{self.org_id}/messages/conversations/{conv_id}/admins/promote",
            json={
                "actor_email": "qa@apexai.io",
                "target_email": "qa@apexai.io",
            }
        )
        self.assertEqual(unauth_promote.status_code, 403)

        # 5. Admin promotes dev to admin
        promote_res = client.post(
            f"/api/organizations/{self.org_id}/messages/conversations/{conv_id}/admins/promote",
            json={
                "actor_email": self.admin_email,
                "target_email": "dev@apexai.io",
            }
        )
        self.assertEqual(promote_res.status_code, 200)
        self.assertIn("dev@apexai.io", promote_res.json()["conversation"]["admins"])

        # 6. Newly promoted admin removes qa
        remove_res = client.post(
            f"/api/organizations/{self.org_id}/messages/conversations/{conv_id}/members/remove",
            json={
                "actor_email": "dev@apexai.io",
                "target_email": "qa@apexai.io",
            }
        )
        self.assertEqual(remove_res.status_code, 200)
        self.assertNotIn("qa@apexai.io", remove_res.json()["conversation"]["members"])

    def test_direct_message_and_attachments(self):
        # 1. Create/get direct chat
        dm_res = client.post(
            f"/api/organizations/{self.org_id}/messages/direct",
            json={
                "user1_email": self.admin_email,
                "user2_email": "dev@apexai.io",
            }
        )
        self.assertEqual(dm_res.status_code, 200)
        dm = dm_res.json()["conversation"]
        conv_id = dm["id"]

        # 2. Upload photo attachment
        fake_image = io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...")
        upload_res = client.post(
            f"/api/organizations/{self.org_id}/messages/upload",
            data={"user_email": self.admin_email},
            files={"file": ("architecture_v2.png", fake_image, "image/png")},
        )
        self.assertEqual(upload_res.status_code, 200)
        att = upload_res.json()["attachment"]
        self.assertEqual(att["file_type"], "photo")
        self.assertIn("/api/organizations/", att["url"])

        # 3. Retrieve uploaded media via GET endpoint
        media_filename = att["saved_name"]
        get_media_res = client.get(f"/api/organizations/{self.org_id}/messages/media/{media_filename}")
        self.assertEqual(get_media_res.status_code, 200)
        self.assertIn("image", get_media_res.headers.get("content-type", ""))

        # 4. Send message with the attachment
        msg_res = client.post(
            f"/api/organizations/{self.org_id}/messages/conversations/{conv_id}/messages",
            json={
                "sender_email": self.admin_email,
                "text": "Review this new pipeline architecture screenshot.",
                "attachments": [att],
            }
        )
        self.assertEqual(msg_res.status_code, 200)
        msg = msg_res.json()["message"]
        msg_id = msg["id"]
        self.assertEqual(len(msg["attachments"]), 1)

        # 5. Toggle reaction
        react_res = client.post(
            f"/api/organizations/{self.org_id}/messages/conversations/{conv_id}/reactions",
            json={
                "message_id": msg_id,
                "emoji": "🚀",
                "user_email": "dev@apexai.io",
            }
        )
        self.assertEqual(react_res.status_code, 200)
        reactions = react_res.json()["reaction"]["reactions"]
        self.assertIn("dev@apexai.io", reactions.get("🚀", []))

        # 6. Fetch conversation message history
        history_res = client.get(f"/api/organizations/{self.org_id}/messages/conversations/{conv_id}/messages?user_email={self.admin_email}")
        self.assertEqual(history_res.status_code, 200)
        msgs = history_res.json()["messages"]
        self.assertTrue(any(m["id"] == msg_id for m in msgs))

    def test_websocket_messaging_connection(self):
        conv_id = f"conv-group-{self.org_id}-general"

        with client.websocket_connect(f"/ws/organizations/{self.org_id}/messages?user_email={self.admin_email}") as ws:
            # 1. Test typing action over WebSocket
            ws.send_json({
                "action": "typing",
                "conversation_id": conv_id,
                "is_typing": True,
                "sender_name": "Admin Tester",
            })
            resp = ws.receive_json()
            self.assertEqual(resp.get("type"), "typing")
            self.assertEqual(resp.get("is_typing"), True)

            # 2. Test send_message action over WebSocket
            ws.send_json({
                "action": "send_message",
                "conversation_id": conv_id,
                "text": "Hello real-time WebSocket messaging!",
                "sender_name": "Admin Tester",
                "attachments": [],
            })
            msg_resp = ws.receive_json()
            self.assertEqual(msg_resp.get("type"), "new_message")
            self.assertEqual(msg_resp.get("conversation_id"), conv_id)
            self.assertEqual(msg_resp["message"]["text"], "Hello real-time WebSocket messaging!")
            created_msg_id = msg_resp["message"]["id"]

            # 3. Test toggle_reaction action over WebSocket
            ws.send_json({
                "action": "toggle_reaction",
                "conversation_id": conv_id,
                "message_id": created_msg_id,
                "emoji": "🎉",
            })
            react_resp = ws.receive_json()
            self.assertEqual(react_resp.get("type"), "reaction_updated")
            self.assertEqual(react_resp.get("conversation_id"), conv_id)
            self.assertIn("🎉", react_resp["reaction"]["reactions"])
            self.assertIn(self.admin_email, react_resp["reaction"]["reactions"]["🎉"])

if __name__ == "__main__":
    unittest.main()
