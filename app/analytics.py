import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

ANALYTICS_FILE = Path(__file__).resolve().parent / "data" / "analytics.json"


class AnalyticsManager:
    """
    Manages organization-specific usage tracking and analytics for AI agents.
    Provides KPIs, 7-day volume trends, channel distribution, per-agent breakdown,
    and recent activity streams strictly scoped to each organization.
    """
    _instance: Optional["AnalyticsManager"] = None

    def __init__(self, storage_path: Path = ANALYTICS_FILE):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {}
        self.load()

    @classmethod
    def get_instance(cls) -> "AnalyticsManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Loads analytics data from disk or seeds initial realistic records."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"⚠️ [AnalyticsManager] Error reading {self.storage_path}: {e}")
                self.data = {}

        if not self.data or "apex-ai-innovations" not in self.data:
            self._seed_default_data()
            self.save()

    def save(self) -> None:
        """Persists analytics to disk."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"⚠️ [AnalyticsManager] Error saving to {self.storage_path}: {e}")

    def _seed_default_data(self) -> None:
        """Seeds realistic historical baseline metrics for default organizations."""
        self.data["apex-ai-innovations"] = {
            "summary": {
                "total_calls": 1284,
                "total_minutes": 142.5,
                "avg_latency_ms": 320,
                "resolution_rate_pct": 98.6,
                "calls_change_pct": "+14.2% vs last week",
                "minutes_change_pct": "+9.8% vs last week"
            },
            "channels": {
                "web_voice": 924,
                "telephony": 360,
                "web_pct": 72.0,
                "telephony_pct": 28.0
            },
            "trend_7d": [
                {"day": "Thu", "date": "Sep 17", "calls": 142, "minutes": 16.4},
                {"day": "Fri", "date": "Sep 18", "calls": 188, "minutes": 21.8},
                {"day": "Sat", "date": "Sep 19", "calls": 95, "minutes": 11.2},
                {"day": "Sun", "date": "Sep 20", "calls": 84, "minutes": 9.5},
                {"day": "Mon", "date": "Sep 21", "calls": 245, "minutes": 28.6},
                {"day": "Tue", "date": "Sep 22", "calls": 268, "minutes": 31.0},
                {"day": "Wed", "date": "Sep 23", "calls": 262, "minutes": 24.0}
            ],
            "agent_metrics": {
                "general-voice-assistant": {
                    "calls": 642,
                    "minutes": 78.4,
                    "avg_latency_ms": 290,
                    "resolution_rate_pct": 99.1
                },
                "tech-assiststant": {
                    "calls": 380,
                    "minutes": 41.2,
                    "avg_latency_ms": 340,
                    "resolution_rate_pct": 98.2
                },
                "market-analysis-pro": {
                    "calls": 142,
                    "minutes": 14.9,
                    "avg_latency_ms": 350,
                    "resolution_rate_pct": 97.8
                },
                "content-writer-bot": {
                    "calls": 76,
                    "minutes": 5.4,
                    "avg_latency_ms": 310,
                    "resolution_rate_pct": 98.5
                },
                "customer-support-ai": {
                    "calls": 44,
                    "minutes": 2.6,
                    "avg_latency_ms": 315,
                    "resolution_rate_pct": 99.4
                }
            },
            "recent_activity": [
                {
                    "id": "act-101",
                    "timestamp": time.time() - 120,
                    "time_ago": "2m ago",
                    "agent_id": "general-voice-assistant",
                    "agent_name": "General Voice Assistant",
                    "channel": "Web Voice",
                    "duration_sec": 48,
                    "latency_ms": 284,
                    "status": "Completed"
                },
                {
                    "id": "act-102",
                    "timestamp": time.time() - 480,
                    "time_ago": "8m ago",
                    "agent_id": "tech-assiststant",
                    "agent_name": "tech_assiststant",
                    "channel": "Telephony",
                    "duration_sec": 112,
                    "latency_ms": 332,
                    "status": "Completed"
                },
                {
                    "id": "act-103",
                    "timestamp": time.time() - 1250,
                    "time_ago": "21m ago",
                    "agent_id": "general-voice-assistant",
                    "agent_name": "General Voice Assistant",
                    "channel": "Web Voice",
                    "duration_sec": 35,
                    "latency_ms": 295,
                    "status": "Completed"
                },
                {
                    "id": "act-104",
                    "timestamp": time.time() - 2400,
                    "time_ago": "40m ago",
                    "agent_id": "market-analysis-pro",
                    "agent_name": "Market Analysis Pro",
                    "channel": "Web Voice",
                    "duration_sec": 64,
                    "latency_ms": 360,
                    "status": "Completed"
                },
                {
                    "id": "act-105",
                    "timestamp": time.time() - 3600,
                    "time_ago": "1h ago",
                    "agent_id": "customer-support-ai",
                    "agent_name": "Customer Support AI",
                    "channel": "Telephony",
                    "duration_sec": 92,
                    "latency_ms": 310,
                    "status": "Completed"
                }
            ]
        }

    def _ensure_org_record(self, org_id: str) -> Dict[str, Any]:
        """Ensures an organization record exists, creating an initial baseline if new."""
        if org_id not in self.data:
            self.data[org_id] = {
                "summary": {
                    "total_calls": 0,
                    "total_minutes": 0.0,
                    "avg_latency_ms": 320,
                    "resolution_rate_pct": 100.0,
                    "calls_change_pct": "+0.0% vs last week",
                    "minutes_change_pct": "+0.0% vs last week"
                },
                "channels": {
                    "web_voice": 0,
                    "telephony": 0,
                    "web_pct": 100.0,
                    "telephony_pct": 0.0
                },
                "trend_7d": [
                    {"day": "Thu", "date": "Sep 17", "calls": 0, "minutes": 0.0},
                    {"day": "Fri", "date": "Sep 18", "calls": 0, "minutes": 0.0},
                    {"day": "Sat", "date": "Sep 19", "calls": 0, "minutes": 0.0},
                    {"day": "Sun", "date": "Sep 20", "calls": 0, "minutes": 0.0},
                    {"day": "Mon", "date": "Sep 21", "calls": 0, "minutes": 0.0},
                    {"day": "Tue", "date": "Sep 22", "calls": 0, "minutes": 0.0},
                    {"day": "Wed", "date": "Sep 23", "calls": 0, "minutes": 0.0}
                ],
                "agent_metrics": {},
                "recent_activity": []
            }
            self.save()
        return self.data[org_id]

    def get_organization_analytics(self, org_id: str, org_agents: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Retrieves analytics strictly scoped to the specified organization.
        Enriches agent breakdown using the actual list of agents registered to this org.
        """
        org_data = self._ensure_org_record(org_id)
        summary = dict(org_data.get("summary", {}))
        channels = dict(org_data.get("channels", {}))
        trend_7d = list(org_data.get("trend_7d", []))
        stored_agent_metrics = org_data.get("agent_metrics", {})
        recent_activity = list(org_data.get("recent_activity", []))

        now = time.time()
        for act in recent_activity:
            elapsed = int(now - act.get("timestamp", now))
            if elapsed < 60:
                act["time_ago"] = "Just now"
            elif elapsed < 3600:
                act["time_ago"] = f"{elapsed // 60}m ago"
            elif elapsed < 86400:
                act["time_ago"] = f"{elapsed // 3600}h ago"
            else:
                act["time_ago"] = f"{elapsed // 86400}d ago"

        agents_breakdown = []
        if org_agents:
            for ag in org_agents:
                ag_id = ag.get("id")
                metrics = stored_agent_metrics.get(ag_id, {
                    "calls": 0,
                    "minutes": 0.0,
                    "avg_latency_ms": 320,
                    "resolution_rate_pct": 100.0
                })
                agents_breakdown.append({
                    "id": ag_id,
                    "name": ag.get("name", ag_id),
                    "role": ag.get("role", "Voice AI"),
                    "status": ag.get("status", "Active"),
                    "is_active": ag.get("is_active", False),
                    "calls": metrics.get("calls", 0),
                    "minutes": metrics.get("minutes", 0.0),
                    "avg_latency_ms": metrics.get("avg_latency_ms", 320),
                    "resolution_rate_pct": metrics.get("resolution_rate_pct", 98.5)
                })

        summary["active_agents_count"] = len(org_agents) if org_agents is not None else 0

        return {
            "org_id": org_id,
            "summary": summary,
            "channels": channels,
            "trend_7d": trend_7d,
            "agents_breakdown": agents_breakdown,
            "recent_activity": recent_activity[:10]
        }

    def record_interaction(
        self,
        org_id: str,
        agent_id: str,
        agent_name: str,
        channel: str = "Web Voice",
        duration_sec: float = 30.0,
        latency_ms: int = 320,
        success: bool = True
    ) -> Dict[str, Any]:
        """
        Records a completed voice/chat interaction for the organization and agent.
        Updates summary totals, channel splits, 7-day trend, and adds to recent activity.
        """
        org_data = self._ensure_org_record(org_id)
        minutes = round(duration_sec / 60.0, 2)

        summary = org_data["summary"]
        current_calls = summary.get("total_calls", 0)
        current_latency = summary.get("avg_latency_ms", 320)
        new_calls = current_calls + 1
        new_latency = int(((current_latency * current_calls) + latency_ms) / new_calls)

        summary["total_calls"] = new_calls
        summary["total_minutes"] = round(summary.get("total_minutes", 0.0) + minutes, 1)
        summary["avg_latency_ms"] = new_latency

        channels = org_data["channels"]
        if channel.lower() == "telephony":
            channels["telephony"] = channels.get("telephony", 0) + 1
        else:
            channels["web_voice"] = channels.get("web_voice", 0) + 1

        total_ch = channels["web_voice"] + channels["telephony"]
        if total_ch > 0:
            channels["web_pct"] = round((channels["web_voice"] / total_ch) * 100, 1)
            channels["telephony_pct"] = round((channels["telephony"] / total_ch) * 100, 1)

        agent_metrics = org_data.setdefault("agent_metrics", {})
        ag_rec = agent_metrics.setdefault(agent_id, {
            "calls": 0,
            "minutes": 0.0,
            "avg_latency_ms": 320,
            "resolution_rate_pct": 98.5
        })
        ag_calls = ag_rec["calls"] + 1
        ag_rec["avg_latency_ms"] = int(((ag_rec["avg_latency_ms"] * ag_rec["calls"]) + latency_ms) / ag_calls)
        ag_rec["calls"] = ag_calls
        ag_rec["minutes"] = round(ag_rec["minutes"] + minutes, 1)

        if org_data["trend_7d"]:
            today_entry = org_data["trend_7d"][-1]
            today_entry["calls"] = today_entry.get("calls", 0) + 1
            today_entry["minutes"] = round(today_entry.get("minutes", 0.0) + minutes, 1)

        activity_item = {
            "id": f"act-{int(time.time() * 1000)}",
            "timestamp": time.time(),
            "time_ago": "Just now",
            "agent_id": agent_id,
            "agent_name": agent_name,
            "channel": channel,
            "duration_sec": int(duration_sec),
            "latency_ms": latency_ms,
            "status": "Completed" if success else "Interrupted"
        }
        org_data["recent_activity"].insert(0, activity_item)
        org_data["recent_activity"] = org_data["recent_activity"][:50]

        self.save()
        return activity_item
