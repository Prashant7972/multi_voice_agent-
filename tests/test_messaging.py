import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.messaging import MessagingManager, detect_file_type
from app.organizations import OrganizationManager

class TestMessagingManager(unittest.TestCase):
    def setUp(self):
        self.org_manager = OrganizationManager.get_instance()
        self.messaging = MessagingManager.get_instance()
        self.org_id = "apex-ai-innovations"
        self.admin_email = "admin@apexai.io"

        # Ensure a second member exists in the organization for testing
        approved_members = [m["email"] for m in self.org_manager.list_members(self.org_id)]
        if "dev@apexai.io" not in approved_members:
            self.org_manager.add_member(
                org_id=self.org_id,
                email="dev@apexai.io",
                name="AI Engineer",
                role="admin",
                adder_email=self.admin_email
            )
        if "analyst@apexai.io" not in approved_members:
            self.org_manager.add_member(
                org_id=self.org_id,
                email="analyst@apexai.io",
                name="Data Analyst",
                role="user",
                adder_email=self.admin_email
            )

    def test_file_type_detection(self):
        self.assertEqual(detect_file_type("report.pdf", "application/pdf"), "document")
        self.assertEqual(detect_file_type("screenshot.PNG", "image/png"), "photo")
        self.assertEqual(detect_file_type("photo.jpeg"), "photo")
        self.assertEqual(detect_file_type("demo.mp4", "video/mp4"), "video")
        self.assertEqual(detect_file_type("recording.webm", "video/webm"), "video")
        self.assertEqual(detect_file_type("data.csv"), "document")

    def test_group_lifecycle_and_admin_security(self):
        # 1. Create group
        group = self.messaging.create_group(
            org_id=self.org_id,
            creator_email=self.admin_email,
            name="ML Research & Development",
            description="Discuss models, pipelines, and voice benchmarks.",
            member_emails=["dev@apexai.io"]
        )
        group_id = group["id"]
        self.assertIn(self.admin_email, group["admins"])
        self.assertIn("dev@apexai.io", group["members"])

        # 2. Non-admin cannot add member
        # analyst is not an admin in this group
        with self.assertRaises(PermissionError):
            self.messaging.add_group_member(
                org_id=self.org_id,
                conversation_id=group_id,
                actor_email="analyst@apexai.io",
                new_member_email="analyst@apexai.io"
            )

        # 3. Admin adds member
        updated = self.messaging.add_group_member(
            org_id=self.org_id,
            conversation_id=group_id,
            actor_email=self.admin_email,
            new_member_email="analyst@apexai.io"
        )
        self.assertIn("analyst@apexai.io", updated["members"])

        # 4. Only org members can be added
        with self.assertRaises(ValueError):
            self.messaging.add_group_member(
                org_id=self.org_id,
                conversation_id=group_id,
                actor_email=self.admin_email,
                new_member_email="stranger@external-domain.com"
            )

        # 5. Non-admin cannot promote admin
        with self.assertRaises(PermissionError):
            self.messaging.promote_group_admin(
                org_id=self.org_id,
                conversation_id=group_id,
                actor_email="analyst@apexai.io",
                target_email="analyst@apexai.io"
            )

        # 6. Admin promotes dev to admin
        promoted = self.messaging.promote_group_admin(
            org_id=self.org_id,
            conversation_id=group_id,
            actor_email=self.admin_email,
            target_email="dev@apexai.io"
        )
        self.assertIn("dev@apexai.io", promoted["admins"])

        # 7. Newly promoted admin can now remove members
        removed = self.messaging.remove_group_member(
            org_id=self.org_id,
            conversation_id=group_id,
            actor_email="dev@apexai.io",
            target_email="analyst@apexai.io"
        )
        self.assertNotIn("analyst@apexai.io", removed["members"])

    def test_direct_messaging(self):
        dm = self.messaging.get_or_create_direct_chat(
            org_id=self.org_id,
            user1_email=self.admin_email,
            user2_email="dev@apexai.io"
        )
        self.assertEqual(dm["type"], "direct")
        self.assertIn(self.admin_email, dm["members"])
        self.assertIn("dev@apexai.io", dm["members"])

        # Re-querying returns same conversation
        dm2 = self.messaging.get_or_create_direct_chat(
            org_id=self.org_id,
            user1_email="dev@apexai.io",
            user2_email=self.admin_email
        )
        self.assertEqual(dm["id"], dm2["id"])

    def test_messages_attachments_reactions(self):
        dm = self.messaging.get_or_create_direct_chat(
            org_id=self.org_id,
            user1_email=self.admin_email,
            user2_email="dev@apexai.io"
        )

        # Save attachment
        att = self.messaging.save_attachment(
            org_id=self.org_id,
            filename="diagram.png",
            file_bytes=b"dummy-png-bytes",
            content_type="image/png"
        )
        self.assertEqual(att["file_type"], "photo")
        self.assertTrue(att["url"].startswith(f"/api/organizations/{self.org_id}/messages/media/"))

        # Save message with attachment
        msg = self.messaging.save_message(
            org_id=self.org_id,
            conversation_id=dm["id"],
            sender_email=self.admin_email,
            sender_name="Platform Administrator",
            text="Here is the architecture snapshot.",
            attachments=[att]
        )
        self.assertEqual(msg["text"], "Here is the architecture snapshot.")
        self.assertEqual(len(msg["attachments"]), 1)

        # Toggle reaction
        res1 = self.messaging.toggle_reaction(
            org_id=self.org_id,
            conversation_id=dm["id"],
            message_id=msg["id"],
            user_email="dev@apexai.io",
            emoji="🔥"
        )
        self.assertIn("dev@apexai.io", res1["reactions"]["🔥"])

        # Toggle off
        res2 = self.messaging.toggle_reaction(
            org_id=self.org_id,
            conversation_id=dm["id"],
            message_id=msg["id"],
            user_email="dev@apexai.io",
            emoji="🔥"
        )
        self.assertNotIn("🔥", res2["reactions"])

    def test_chat_deletion_and_clear_permissions(self):
        # 1. Create a conversation
        group = self.messaging.create_group(
            org_id=self.org_id,
            creator_email=self.admin_email,
            name="Security Review Group",
            member_emails=["dev@apexai.io", "analyst@apexai.io"]
        )
        group_id = group["id"]

        # 2. Add message from dev
        msg = self.messaging.save_message(
            org_id=self.org_id,
            conversation_id=group_id,
            sender_email="dev@apexai.io",
            sender_name="AI Engineer",
            text="Initial security draft"
        )
        msg_id = msg["id"]

        # 3. Regular user (analyst) cannot clear conversation history
        with self.assertRaises(PermissionError):
            self.messaging.clear_conversation(
                org_id=self.org_id,
                conversation_id=group_id,
                actor_email="analyst@apexai.io"
            )

        # 4. Regular user (analyst) cannot delete someone else's message
        with self.assertRaises(PermissionError):
            self.messaging.delete_message(
                org_id=self.org_id,
                conversation_id=group_id,
                message_id=msg_id,
                actor_email="analyst@apexai.io"
            )

        # 5. Original author (dev) CAN delete their own message
        res_del = self.messaging.delete_message(
            org_id=self.org_id,
            conversation_id=group_id,
            message_id=msg_id,
            actor_email="dev@apexai.io"
        )
        self.assertEqual(res_del["message_id"], msg_id)

        # 6. Admin CAN clear conversation history
        res_clear = self.messaging.clear_conversation(
            org_id=self.org_id,
            conversation_id=group_id,
            actor_email=self.admin_email
        )
        self.assertEqual(res_clear["cleared_by"], self.admin_email)

if __name__ == "__main__":
    unittest.main()

