import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import atomic_save_json

AGENTS_FILE = Path(__file__).resolve().parent / "data" / "agents.json"

DEFAULT_SYSTEM_TEMPLATE = (
    "You are {name}, a specialized {role}.\n"
    "Your primary goals and objectives are: {goals}.\n\n"
    "CRITICAL CONVERSATIONAL VOICE INSTRUCTIONS:\n"
    "- You are speaking out loud to the user over a real-time voice channel.\n"
    "- Always answer in 1 to 2 concise, clear, and direct spoken sentences.\n"
    "- NEVER use markdown formatting, asterisks, bold text, hashes, bullet points, or code blocks.\n"
    "- Sound natural, engaging, and professional.\n"
    "- When user speaks or asks questions, address them directly."
)


def slugify(text: str) -> str:
    """Converts a title or name into a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-") or "custom-agent"


class AgentManager:
    """
    Manages persistence, lifecycle, and dynamic persona configuration for AI Agents.
    """
    _instance: Optional["AgentManager"] = None

    def __init__(self, storage_path: Path = AGENTS_FILE):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.agents: List[Dict[str, Any]] = []
        self.load()

    @classmethod
    def get_instance(cls) -> "AgentManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Loads agents from disk, or seeds default agents if absent."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.agents = json.load(f)
                    for ag in self.agents:
                        if "org_id" not in ag:
                            ag["org_id"] = "apex-ai-innovations"
            except Exception as e:
                print(f"⚠️ [AgentManager] Error reading {self.storage_path}: {e}")
                self.agents = []

        if not self.agents:
            self.agents = [
                {
                    "id": "general-voice-assistant",
                    "name": "General Voice Assistant",
                    "role": "Personal Voice AI",
                    "status": "Active",
                    "last_updated": "Just now",
                    "updated_at": time.time(),
                    "goals": "Deliver lightning-fast, natural spoken answers with neural noise-filtered barge-in and telephone support.",
                    "system_prompt": (
                        "You are General Voice Assistant, a friendly, ultra-fast spoken conversational assistant. "
                        "Always respond in 1-2 concise sentences suitable for voice text-to-speech. "
                        "Never use markdown formatting, bullet points, or lists. Be warm, accurate, and direct."
                    ),
                    "capabilities": [
                        "Live Voice Interaction",
                        "Twilio Telephony Calling",
                        "Neural VAD Noise Filter",
                        "Conversational Barge-In"
                    ],
                    "is_active": True,
                },
                {
                    "id": "market-analysis-pro",
                    "name": "Market Analysis Pro",
                    "role": "Data Scientist",
                    "status": "Active",
                    "last_updated": "2m ago",
                    "updated_at": time.time() - 120,
                    "goals": "Analyze market trends, forecast sales, optimize pricing.",
                    "system_prompt": (
                        "You are Market Analysis Pro, an expert data scientist and quantitative analyst. "
                        "Your goal is to analyze market trends, forecast sales, and optimize pricing. "
                        "Speak with sharp, data-driven clarity in 1-2 concise sentences for voice conversation. "
                        "Avoid jargon overload and deliver actionable insights."
                    ),
                    "capabilities": [
                        "Web Search",
                        "Data Extraction",
                        "Report Generation",
                        "Live Voice Interaction",
                        "Twilio Telephony Calling"
                    ],
                    "is_active": False,
                },
                {
                    "id": "content-writer-bot",
                    "name": "Content Writer Bot",
                    "role": "Copywriter",
                    "status": "Active",
                    "last_updated": "15m ago",
                    "updated_at": time.time() - 900,
                    "goals": "Draft compelling copy, refine storytelling, and polish social media messaging.",
                    "system_prompt": (
                        "You are Content Writer Bot, a creative and persuasive copywriter. "
                        "Your goal is to draft compelling copy, engage audiences, and refine brand messaging. "
                        "When speaking aloud, craft punchy, memorable, and energetic responses in 1-2 spoken sentences."
                    ),
                    "capabilities": [
                        "Web Search",
                        "Extraction",
                        "Report Generation",
                        "Live Voice Interaction",
                        "Twilio Telephony Calling"
                    ],
                    "is_active": False,
                },
                {
                    "id": "customer-support-ai",
                    "name": "Customer Support AI",
                    "role": "Support Specialist",
                    "status": "Active",
                    "last_updated": "1h ago",
                    "updated_at": time.time() - 3600,
                    "goals": "Resolve customer issues, troubleshoot product questions, and provide empathetic assistance.",
                    "system_prompt": (
                        "You are Customer Support AI, a professional, empathetic support specialist. "
                        "Your goal is to assist users warmly, solve their problems patiently, and guide them through step-by-step troubleshooting. "
                        "Always answer in 1-2 concise, clear sentences suitable for voice conversation."
                    ),
                    "capabilities": [
                        "Web Search",
                        "Data Extraction",
                        "Report Generation",
                        "Live Voice Interaction",
                        "Twilio Telephony Calling"
                    ],
                    "is_active": False,
                },
            ]
            for ag in self.agents:
                if "org_id" not in ag:
                    ag["org_id"] = "apex-ai-innovations"
            self.save()

    def save(self) -> None:
        """Persists agent registry to disk atomically."""
        try:
            atomic_save_json(self.storage_path, self.agents)
        except Exception as e:
            print(f"⚠️ [AgentManager] Error saving to {self.storage_path}: {e}")

    def list_agents(self, org_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns list of all available agents, optionally filtered by org_id."""
        if org_id:
            org_id_clean = org_id.strip().lower()
            return [
                ag for ag in self.agents
                if (ag.get("org_id") or "apex-ai-innovations").strip().lower() == org_id_clean
            ]
        return self.agents

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Finds an agent by id."""
        for ag in self.agents:
            if ag.get("id") == agent_id:
                return ag
        return None

    def get_active_agent(self) -> Dict[str, Any]:
        """Returns the currently active agent, or the first available."""
        for ag in self.agents:
            if ag.get("is_active"):
                return ag
        if self.agents:
            self.agents[0]["is_active"] = True
            self.save()
            return self.agents[0]
        # Fallback if empty
        self.load()
        return self.agents[0]

    def set_active_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Marks the specified agent as active, deactivating others."""
        target = None
        for ag in self.agents:
            if ag.get("id") == agent_id:
                ag["is_active"] = True
                ag["status"] = "Active"
                target = ag
            else:
                ag["is_active"] = False
        if target:
            self.save()
        return target

    def create_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a new agent with auto-generated slug and system prompt."""
        name = data.get("name", "").strip() or "New Custom Agent"
        role = data.get("role", "").strip() or "Specialized Assistant"
        goals = data.get("goals", "").strip() or "Assist users and perform tasks efficiently."
        org_id = data.get("org_id", "").strip() or "apex-ai-innovations"
        
        # Base ID
        base_id = slugify(name)
        agent_id = base_id
        counter = 1
        existing_ids = {ag.get("id") for ag in self.agents}
        while agent_id in existing_ids:
            agent_id = f"{base_id}-{counter}"
            counter += 1

        # System prompt
        system_prompt = data.get("system_prompt", "").strip()
        if not system_prompt:
            system_prompt = DEFAULT_SYSTEM_TEMPLATE.format(name=name, role=role, goals=goals)

        capabilities = data.get("capabilities", [])
        if not capabilities:
            capabilities = ["Live Voice Interaction", "Twilio Telephony Calling"]

        new_agent = {
            "id": agent_id,
            "org_id": org_id,
            "name": name,
            "role": role,
            "status": "Active",
            "last_updated": "Just now",
            "updated_at": time.time(),
            "goals": goals,
            "system_prompt": system_prompt,
            "capabilities": capabilities,
            "is_active": False,
        }

        self.agents.insert(0, new_agent)
        self.save()
        return new_agent

    def update_agent(self, agent_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Updates an existing agent's attributes."""
        agent = self.get_agent(agent_id)
        if not agent:
            return None

        if "name" in data and data["name"].strip():
            agent["name"] = data["name"].strip()
        if "role" in data and data["role"].strip():
            agent["role"] = data["role"].strip()
        if "goals" in data and data["goals"].strip():
            agent["goals"] = data["goals"].strip()
        if "system_prompt" in data and data["system_prompt"].strip():
            agent["system_prompt"] = data["system_prompt"].strip()
        elif "goals" in data or "role" in data or "name" in data:
            agent["system_prompt"] = DEFAULT_SYSTEM_TEMPLATE.format(
                name=agent["name"], role=agent["role"], goals=agent["goals"]
            )
        if "capabilities" in data and isinstance(data["capabilities"], list):
            agent["capabilities"] = data["capabilities"]
        if "status" in data:
            agent["status"] = data["status"]

        agent["last_updated"] = "Just now"
        agent["updated_at"] = time.time()
        self.save()
        return agent

    def delete_agent(self, agent_id: str) -> bool:
        """Deletes an agent by ID (cannot delete if it is the only agent)."""
        if len(self.agents) <= 1:
            return False
        
        target = self.get_agent(agent_id)
        if not target:
            return False

        was_active = target.get("is_active", False)
        self.agents = [ag for ag in self.agents if ag.get("id") != agent_id]
        if was_active and self.agents:
            self.agents[0]["is_active"] = True

        self.save()
        return True
