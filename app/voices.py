"""
Organization Custom Voice Personas Manager
Provides persistence, lifecycle, and organization-scoped management of custom voice personas.
"""
import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import atomic_save_json

logger = logging.getLogger("voice_agent.voices")

VOICES_FILE = Path(__file__).resolve().parent / "data" / "voices.json"


def slugify(text: str) -> str:
    """Converts a title or name into a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-") or "custom-voice"


class VoiceManager:
    """
    Manages custom-crafted voice personas scoped independently per organization.
    """
    _instance: Optional["VoiceManager"] = None

    def __init__(self, storage_path: Path = VOICES_FILE):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.voices: List[Dict[str, Any]] = []
        self.load()

    @classmethod
    def get_instance(cls) -> "VoiceManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Loads custom voices from disk, or seeds initial sample voice if absent."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.voices = json.load(f)
            except Exception as e:
                logger.error("Error reading %s: %s", self.storage_path, e)
                self.voices = []

        if not self.voices:
            # Seed default custom personas for default organization
            self.voices = [
                {
                    "id": "voice-apex-maya-caregiver",
                    "name": "Maya - Empathetic Care",
                    "org_id": "apex-ai-innovations",
                    "gender": "female",
                    "age_group": "young_adult",
                    "personality": "empathetic_caregiver",
                    "personality_label": "Empathetic Caregiver",
                    "language": "en",
                    "language_name": "English (US)",
                    "base_voice_id": "en-US-AvaNeural",
                    "voice_model": "en-US-AvaNeural",
                    "pitch": "+6Hz",
                    "rate": "-5%",
                    "sample_phrase": "Hello! I am here to assist and support you today. How are you doing?",
                    "notes": "Gentle, supportive, active listener designed for patient support",
                    "created_at": time.time() - 86400,
                    "created_by": "admin@apexai.io",
                },
                {
                    "id": "voice-apex-priya-hindi",
                    "name": "Priya - Tech Specialist (Hindi/English)",
                    "org_id": "apex-ai-innovations",
                    "gender": "female",
                    "age_group": "young_adult",
                    "personality": "professional_crisp",
                    "personality_label": "Professional & Crisp",
                    "language": "hi",
                    "language_name": "Hindi (हिन्दी)",
                    "base_voice_id": "hi-IN-SwaraNeural",
                    "voice_model": "hi-IN-SwaraNeural",
                    "pitch": "+4Hz",
                    "rate": "+5%",
                    "sample_phrase": "नमस्ते! मैं आपकी तकनीकी सलाहकार हूँ। आज हम किस समस्या का समाधान करेंगे?",
                    "notes": "Articulate, goal-oriented technical specialist",
                    "created_at": time.time() - 43200,
                    "created_by": "admin@apexai.io",
                },
                {
                    "id": "voice-apex-marcus-advisor",
                    "name": "Marcus - Executive Counselor",
                    "org_id": "apex-ai-innovations",
                    "gender": "aged",
                    "age_group": "senior",
                    "personality": "calm_advisor",
                    "personality_label": "Calm Advisor",
                    "language": "en",
                    "language_name": "English (US)",
                    "base_voice_id": "en-US-AndrewNeural",
                    "voice_model": "en-US-AndrewNeural",
                    "pitch": "-12Hz",
                    "rate": "-10%",
                    "sample_phrase": "Welcome. Let us review your strategy carefully and plan our next steps.",
                    "notes": "Distinguished, warm elder wisdom suited for leadership consultation",
                    "created_at": time.time() - 10800,
                    "created_by": "admin@apexai.io",
                },
            ]
            self.save()

    def save(self) -> None:
        """Persists custom voice personas atomically."""
        try:
            atomic_save_json(self.storage_path, self.voices)
        except Exception as e:
            logger.error("Error saving custom voices to %s: %s", self.storage_path, e)

    def list_voices(self, org_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns custom voices for a specific organization (or all if org_id is None)."""
        if org_id:
            clean_org = org_id.strip().lower()
            return [v for v in self.voices if (v.get("org_id") or "apex-ai-innovations").strip().lower() == clean_org]
        return self.voices

    def get_voice(self, org_id: str, voice_id: str) -> Optional[Dict[str, Any]]:
        """Finds a custom voice persona by ID within an organization."""
        clean_org = (org_id or "").strip().lower()
        clean_id = (voice_id or "").strip()
        for v in self.voices:
            if v.get("id") == clean_id and (v.get("org_id") or "apex-ai-innovations").strip().lower() == clean_org:
                return v
        return None

    def create_voice(self, org_id: str, data: Dict[str, Any], created_by: str = "admin@apexai.io") -> Dict[str, Any]:
        """Creates and saves a new custom voice persona for an organization."""
        clean_org = (org_id or "apex-ai-innovations").strip().lower()
        name = (data.get("name") or "Custom Voice Persona").strip()
        slug = slugify(name)
        voice_id = f"voice-{clean_org}-{slug}-{int(time.time() * 1000)}"

        gender = (data.get("gender") or "female").strip().lower()
        age_group = (data.get("age_group") or "young_adult").strip().lower()
        personality = (data.get("personality") or "friendly_warm").strip().lower()
        personality_label = data.get("personality_label") or personality.replace("_", " ").title()
        language = (data.get("language") or "en").strip().lower()
        language_name = data.get("language_name") or ("Hindi" if language == "hi" else ("Marathi" if language == "mr" else "English"))
        base_voice_id = data.get("base_voice_id") or data.get("voice_model") or "en-US-AvaNeural"
        pitch = data.get("pitch") or "+0Hz"
        rate = data.get("rate") or "+0%"
        sample_phrase = data.get("sample_phrase") or "Hello! This is a test audition of my voice persona."
        notes = data.get("notes") or f"Customized in Voice Studio for {clean_org}"

        new_voice: Dict[str, Any] = {
            "id": voice_id,
            "name": name,
            "org_id": clean_org,
            "gender": gender,
            "age_group": age_group,
            "personality": personality,
            "personality_label": personality_label,
            "language": language,
            "language_name": language_name,
            "base_voice_id": base_voice_id,
            "voice_model": base_voice_id,
            "pitch": pitch,
            "rate": rate,
            "sample_phrase": sample_phrase,
            "notes": notes,
            "created_at": time.time(),
            "created_by": created_by,
        }

        self.voices.insert(0, new_voice)
        self.save()
        logger.info("Created custom voice persona %s (%s) for org %s", name, voice_id, clean_org)
        return new_voice

    def delete_voice(self, org_id: str, voice_id: str) -> bool:
        """Deletes a single custom voice persona by ID within an organization."""
        clean_org = (org_id or "").strip().lower()
        clean_id = (voice_id or "").strip()
        before_count = len(self.voices)
        self.voices = [
            v for v in self.voices
            if not (v.get("id") == clean_id and (v.get("org_id") or "apex-ai-innovations").strip().lower() == clean_org)
        ]
        if len(self.voices) < before_count:
            self.save()
            logger.info("Deleted custom voice persona %s from org %s", clean_id, clean_org)
            return True
        return False

    def delete_voices_batch(self, org_id: str, voice_ids: List[str]) -> int:
        """Deletes multiple custom voice personas by IDs within an organization."""
        clean_org = (org_id or "").strip().lower()
        ids_to_delete = set((vid or "").strip() for vid in voice_ids if vid)
        if not ids_to_delete:
            return 0

        before_count = len(self.voices)
        self.voices = [
            v for v in self.voices
            if not (v.get("id") in ids_to_delete and (v.get("org_id") or "apex-ai-innovations").strip().lower() == clean_org)
        ]
        deleted_count = before_count - len(self.voices)
        if deleted_count > 0:
            self.save()
            logger.info("Batch deleted %d custom voices from org %s", deleted_count, clean_org)
        return deleted_count
