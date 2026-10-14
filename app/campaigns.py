import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.contacts import ContactsManager
from app.twilio_handler import TwilioService

CAMPAIGNS_FILE = Path(__file__).resolve().parent / "data" / "campaigns.json"


class CampaignManager:
    """
    Manages outbound voice calling campaigns:
    - Scoped by organization
    - Bound strictly to a designated AI Agent
    - Targets a named Contact Group
    - Concurrent background call worker using semaphore throttling
    - Live tracking of progress, active lines, and call logs
    """
    _instance: Optional["CampaignManager"] = None

    def __init__(self, storage_path: Path = CAMPAIGNS_FILE):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.campaigns: List[Dict[str, Any]] = []
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._pause_flags: Dict[str, bool] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
        self.load()

    @classmethod
    def get_instance(cls) -> "CampaignManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Loads campaigns from JSON, seeding default sample campaign if absent."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.campaigns = data.get("campaigns", [])
            except Exception as e:
                print(f"⚠️ [CampaignManager] Error reading {self.storage_path}: {e}")
                self.campaigns = []

        if not self.campaigns:
            self._seed_defaults()

    def _seed_defaults(self) -> None:
        """Seeds default sample campaign for apex-ai-innovations."""
        now = time.time()
        self.campaigns = [
            {
                "id": "cmp-q4-enterprise-launch",
                "org_id": "apex-ai-innovations",
                "name": "Q4 Enterprise Voice AI Outreach",
                "description": "Outbound outreach to introduce enterprise clients to low-latency local voice agents and schedule custom architecture reviews.",
                "scheduled_date": "2026-10-05T14:00",
                "group_id": "grp-outreach-q4",
                "group_name": "Q4 Enterprise Outreach",
                "agent_id": "general-voice-assistant",
                "agent_name": "Apex Voice Assistant",
                "concurrency_limit": 3,
                "status": "scheduled",
                "created_at": now - 7200,
                "created_by": "admin@apexai.io",
                "started_at": None,
                "completed_at": None,
                "stats": {
                    "total_contacts": 2,
                    "calls_completed": 0,
                    "calls_in_progress": 0,
                    "calls_failed": 0,
                    "calls_pending": 2,
                },
                "call_logs": [
                    {
                        "call_id": "call-seed-1",
                        "contact_id": "cnt-1001",
                        "contact_name": "Sarah Connor",
                        "phone": "+14155550199",
                        "agent_id": "general-voice-assistant",
                        "status": "pending",
                        "duration_sec": 0,
                        "call_sid": None,
                        "timestamp": now - 7200,
                        "summary": "Scheduled for outbound batch dispatch",
                    },
                    {
                        "call_id": "call-seed-2",
                        "contact_id": "cnt-1002",
                        "contact_name": "Alexander Hayes",
                        "phone": "+442079460912",
                        "agent_id": "general-voice-assistant",
                        "status": "pending",
                        "duration_sec": 0,
                        "call_sid": None,
                        "timestamp": now - 7200,
                        "summary": "Scheduled for outbound batch dispatch",
                    },
                ],
            }
        ]
        self.save()

    def save(self) -> None:
        """Persists campaigns to JSON."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"campaigns": self.campaigns}, f, indent=2)
        except Exception as e:
            print(f"⚠️ [CampaignManager] Error saving to {self.storage_path}: {e}")

    # --- CRUD Methods ---

    def list_campaigns(self, org_id: str) -> List[Dict[str, Any]]:
        """Lists all campaigns for an organization."""
        org_id_clean = org_id.strip().lower()
        res = [c for c in self.campaigns if c.get("org_id", "").strip().lower() == org_id_clean]
        res.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return res

    def get_campaign(self, org_id: str, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a single campaign by ID."""
        org_id_clean = org_id.strip().lower()
        for c in self.campaigns:
            if c.get("org_id", "").strip().lower() == org_id_clean and c.get("id") == campaign_id:
                return c
        return None

    def create_campaign(
        self,
        org_id: str,
        name: str,
        description: str,
        scheduled_date: str,
        group_id: str,
        agent_id: str,
        agent_name: str = "Voice Agent",
        concurrency_limit: int = 3,
        created_by: str = "",
    ) -> Dict[str, Any]:
        """Creates a new outbound calling campaign with initial contact logs."""
        if not name or not name.strip():
            raise ValueError("Campaign name is required.")
        if not group_id:
            raise ValueError("Target contact group is required.")
        if not agent_id:
            raise ValueError("Designated AI Agent is required.")

        org_id_clean = org_id.strip().lower()
        contacts_mgr = ContactsManager.get_instance()
        group = contacts_mgr.get_group(org_id_clean, group_id)
        if not group:
            if group_id in ["default_group", "general", "all"] or not contacts_mgr.list_groups(org_id_clean):
                existing_groups = contacts_mgr.list_groups(org_id_clean)
                if existing_groups:
                    group = existing_groups[0]
                    group_id = group["id"]
                else:
                    group = contacts_mgr.create_group(org_id_clean, name="General Contacts", description="Default contact list for outbound campaigns", created_by=created_by)
                    group_id = group["id"]
            else:
                raise ValueError(f"Contact group '{group_id}' not found.")

        group_name = group.get("name", "Contact Group")
        contacts = contacts_mgr.list_contacts(org_id_clean, group_id=group_id)
        if not contacts and group_id in ["default_group", "general", group.get("id")]:
            all_org_contacts = contacts_mgr.list_contacts(org_id_clean)
            if all_org_contacts:
                contacts = all_org_contacts

        campaign_id = f"cmp-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
        now = time.time()

        # Build initial call logs for each contact in group
        call_logs = []
        for c in contacts:
            call_logs.append({
                "call_id": f"call-{int(time.time() * 1000)}-{random.randint(100, 999)}",
                "contact_id": c.get("id"),
                "contact_name": c.get("name"),
                "phone": c.get("full_phone") or c.get("phone"),
                "agent_id": agent_id,
                "status": "pending",
                "duration_sec": 0,
                "call_sid": None,
                "timestamp": now,
                "summary": "Queued for campaign launch",
            })

        concurrency = max(1, min(10, int(concurrency_limit or 3)))

        new_campaign = {
            "id": campaign_id,
            "org_id": org_id_clean,
            "name": name.strip(),
            "description": description.strip() if description else "",
            "scheduled_date": scheduled_date.strip() if scheduled_date else "",
            "group_id": group_id,
            "group_name": group_name,
            "agent_id": agent_id,
            "agent_name": agent_name.strip() or "Voice Agent",
            "concurrency_limit": concurrency,
            "status": "scheduled",
            "created_at": now,
            "created_by": created_by.strip() or "system",
            "started_at": None,
            "completed_at": None,
            "stats": {
                "total_contacts": len(contacts),
                "calls_completed": 0,
                "calls_in_progress": 0,
                "calls_failed": 0,
                "calls_pending": len(contacts),
            },
            "call_logs": call_logs,
        }

        self.campaigns.append(new_campaign)
        self.save()
        return new_campaign

    def update_campaign(
        self,
        org_id: str,
        campaign_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scheduled_date: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        concurrency_limit: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Updates campaign configuration if not currently running."""
        campaign = self.get_campaign(org_id, campaign_id)
        if not campaign:
            return None

        if name is not None and name.strip():
            campaign["name"] = name.strip()
        if description is not None:
            campaign["description"] = description.strip()
        if scheduled_date is not None:
            campaign["scheduled_date"] = scheduled_date.strip()
        if agent_id is not None and agent_id.strip():
            campaign["agent_id"] = agent_id.strip()
            # Also update agent_id on pending logs
            for log in campaign.get("call_logs", []):
                if log.get("status") == "pending":
                    log["agent_id"] = agent_id.strip()
        if agent_name is not None and agent_name.strip():
            campaign["agent_name"] = agent_name.strip()
        if concurrency_limit is not None:
            campaign["concurrency_limit"] = max(1, min(10, int(concurrency_limit)))

        self.save()
        return campaign

    def delete_campaign(self, org_id: str, campaign_id: str) -> bool:
        """Deletes a campaign and terminates any background running worker."""
        org_id_clean = org_id.strip().lower()
        if campaign_id in self._running_tasks:
            task = self._running_tasks[campaign_id]
            task.cancel()
            del self._running_tasks[campaign_id]

        initial_len = len(self.campaigns)
        self.campaigns = [c for c in self.campaigns if not (c.get("org_id", "").strip().lower() == org_id_clean and c.get("id") == campaign_id)]
        if len(self.campaigns) < initial_len:
            self.save()
            return True
        return False

    # --- Concurrent Execution Engine ---

    def launch_campaign(self, org_id: str, campaign_id: str, host: Optional[str] = None) -> Dict[str, Any]:
        """
        Launches concurrent outbound calling for the specified campaign.
        Runs background asynchronous workers strictly bound to the campaign's designated agent.
        """
        campaign = self.get_campaign(org_id, campaign_id)
        if not campaign:
            raise ValueError(f"Campaign '{campaign_id}' not found.")

        if campaign.get("status") == "in_progress":
            return {"status": "in_progress", "message": "Campaign is already running."}

        # Refresh contacts in group in case contacts were added since creation
        contacts_mgr = ContactsManager.get_instance()
        current_contacts = contacts_mgr.list_contacts(org_id, group_id=campaign.get("group_id"))
        existing_contact_ids = {log.get("contact_id") for log in campaign.get("call_logs", [])}

        for c in current_contacts:
            if c.get("id") not in existing_contact_ids:
                campaign["call_logs"].append({
                    "call_id": f"call-{int(time.time() * 1000)}-{random.randint(100, 999)}",
                    "contact_id": c.get("id"),
                    "contact_name": c.get("name"),
                    "phone": c.get("full_phone") or c.get("phone"),
                    "agent_id": campaign.get("agent_id"),
                    "status": "pending",
                    "duration_sec": 0,
                    "call_sid": None,
                    "timestamp": time.time(),
                    "summary": "Queued for dispatch",
                })

        # Reset any failed or pending logs if re-running
        for log in campaign.get("call_logs", []):
            if log.get("status") in ["failed", "pending"]:
                log["status"] = "pending"

        self._recalc_stats(campaign)
        campaign["status"] = "in_progress"
        campaign["started_at"] = time.time()
        self._pause_flags[campaign_id] = False
        self.save()

        # Launch background runner task
        loop = asyncio.get_event_loop()
        task = loop.create_task(self._run_campaign_worker(campaign_id, host))
        self._running_tasks[campaign_id] = task

        return {
            "status": "in_progress",
            "message": f"Campaign '{campaign.get('name')}' launched with concurrency {campaign.get('concurrency_limit')}.",
            "campaign": campaign,
        }

    def pause_campaign(self, org_id: str, campaign_id: str) -> Dict[str, Any]:
        """Pauses a running campaign."""
        campaign = self.get_campaign(org_id, campaign_id)
        if not campaign:
            raise ValueError(f"Campaign '{campaign_id}' not found.")

        self._pause_flags[campaign_id] = True
        campaign["status"] = "paused"
        self.save()
        return {"status": "paused", "message": f"Campaign '{campaign.get('name')}' paused."}

    def _recalc_stats(self, campaign: Dict[str, Any]) -> None:
        """Recalculates progress stats from call logs."""
        logs = campaign.get("call_logs", [])
        total = len(logs)
        completed = sum(1 for l in logs if l.get("status") == "completed")
        in_progress = sum(1 for l in logs if l.get("status") in ["calling", "in_progress"])
        failed = sum(1 for l in logs if l.get("status") == "failed")
        pending = sum(1 for l in logs if l.get("status") == "pending")

        campaign["stats"] = {
            "total_contacts": total,
            "calls_completed": completed,
            "calls_in_progress": in_progress,
            "calls_failed": failed,
            "calls_pending": pending,
        }

    async def _run_campaign_worker(self, campaign_id: str, host: Optional[str] = None) -> None:
        """
        Background task worker that executes outbound calls with concurrency throttle.
        Strictly connects each call using the designated agent_id.
        """
        campaign = None
        for c in self.campaigns:
            if c.get("id") == campaign_id:
                campaign = c
                break

        if not campaign:
            return

        concurrency = campaign.get("concurrency_limit", 3)
        semaphore = asyncio.Semaphore(concurrency)
        twilio_svc = TwilioService()
        agent_id = campaign.get("agent_id", "general-voice-assistant")
        agent_name = campaign.get("agent_name", "Voice Agent")

        async def _call_contact(log_entry: Dict[str, Any]):
            async with semaphore:
                # Check pause flag
                if self._pause_flags.get(campaign_id, False):
                    return

                phone = log_entry.get("phone", "")
                log_entry["status"] = "calling"
                log_entry["timestamp"] = time.time()
                self._recalc_stats(campaign)
                self.save()

                # Step 1: Initiate call (live Twilio or mock simulated)
                if twilio_svc.is_configured():
                    # Pass designated agent_id to make_call so the TwiML webhook connects to this exact agent
                    res = twilio_svc.make_call(to_phone=phone, host=host, agent_id=agent_id, campaign_id=campaign_id)
                    if res.get("status") == "success":
                        log_entry["call_sid"] = res.get("call_sid")
                        log_entry["status"] = "in_progress"
                        log_entry["summary"] = f"Ringing... Connecting caller to {agent_name}."
                        self._recalc_stats(campaign)
                        self.save()
                        # Wait for call duration to progress or status_callback to finalize
                        wait_start = time.time()
                        while time.time() - wait_start < 120:
                            await asyncio.sleep(2.0)
                            if self._pause_flags.get(campaign_id, False):
                                break
                            if log_entry.get("status") in ["completed", "failed"]:
                                break
                        # If completed without explicit duration callback, set elapsed duration
                        if log_entry.get("status") == "in_progress":
                            log_entry["status"] = "completed"
                            log_entry["duration_sec"] = max(1, int(time.time() - wait_start))
                            log_entry["summary"] = f"Call completed. Spoke with {agent_name}."
                    else:
                        log_entry["status"] = "failed"
                        log_entry["summary"] = res.get("message", "Call initiation failed.")
                else:
                    # Simulated mock outbound calling with realistic progression
                    await asyncio.sleep(1.0)
                    if self._pause_flags.get(campaign_id, False):
                        log_entry["status"] = "pending"
                        return

                    log_entry["status"] = "in_progress"
                    log_entry["call_sid"] = f"SIM-{random.randint(100000, 999999)}"
                    log_entry["summary"] = f"Connected to {agent_name} (Simulated). Speaking with contact..."
                    self._recalc_stats(campaign)
                    self.save()

                    # Realistic call duration in mock mode
                    call_duration = random.randint(25, 75)
                    await asyncio.sleep(2.5)  # brief pause for live UI observation

                    if self._pause_flags.get(campaign_id, False):
                        log_entry["status"] = "pending"
                        return

                    log_entry["status"] = "completed"
                    log_entry["duration_sec"] = call_duration
                    log_entry["summary"] = f"Call completed successfully. Contact spoke with {agent_name}."

                self._recalc_stats(campaign)
                self.save()

        # Collect pending call logs
        pending_logs = [l for l in campaign.get("call_logs", []) if l.get("status") == "pending"]

        tasks = [_call_contact(log) for log in pending_logs]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Finalize campaign status
        self._recalc_stats(campaign)
        if not self._pause_flags.get(campaign_id, False):
            if campaign["stats"]["calls_pending"] == 0 and campaign["stats"]["calls_in_progress"] == 0:
                campaign["status"] = "completed"
                campaign["completed_at"] = time.time()

        self.save()
        if campaign_id in self._running_tasks:
            del self._running_tasks[campaign_id]

    def update_call_status(
        self,
        campaign_id: Optional[str],
        call_sid: str,
        call_status: str,
        duration_sec: int = 0,
        summary: Optional[str] = None,
    ) -> bool:
        """Updates a call log entry from a Twilio status callback or webhook event."""
        if not call_sid:
            return False
        call_status_lower = call_status.strip().lower()
        target_campaigns = [c for c in self.campaigns if c.get("id") == campaign_id] if campaign_id else self.campaigns
        for c in target_campaigns:
            for log in c.get("call_logs", []):
                if log.get("call_sid") == call_sid:
                    if call_status_lower in ["completed"]:
                        log["status"] = "completed"
                        if duration_sec > 0:
                            log["duration_sec"] = duration_sec
                        log["summary"] = summary or f"Call completed successfully ({log.get('duration_sec', duration_sec)}s)."
                    elif call_status_lower in ["busy", "no-answer", "no_answer", "canceled", "failed"]:
                        log["status"] = "failed"
                        log["summary"] = summary or f"Call outcome: {call_status_lower.replace('-', ' ')}."
                    elif call_status_lower in ["in-progress", "in_progress", "answered"]:
                        log["status"] = "in_progress"
                        log["summary"] = summary or f"In call with {c.get('agent_name', 'Voice Agent')}."
                    elif call_status_lower in ["ringing", "initiated", "queued"]:
                        log["status"] = "calling"
                        log["summary"] = summary or "Phone line ringing..."

                    self._recalc_stats(c)
                    if c["stats"]["calls_pending"] == 0 and c["stats"]["calls_in_progress"] == 0:
                        c["status"] = "completed"
                        c["completed_at"] = time.time()
                    self.save()
                    return True
        return False

    async def check_and_trigger_scheduled(self) -> List[str]:
        """
        Scans campaigns with status='scheduled' and auto-launches any whose
        scheduled_date has arrived.
        """
        from datetime import datetime
        now = datetime.now()
        triggered = []

        for c in list(self.campaigns):
            if c.get("status") != "scheduled":
                continue
            sched_str = (c.get("scheduled_date") or "").strip()
            if not sched_str:
                continue
            try:
                # Support standard ISO datetime strings (e.g. 2026-10-01T14:23)
                sched_dt = datetime.fromisoformat(sched_str)
                now_comp = datetime.now(sched_dt.tzinfo) if sched_dt.tzinfo else now
                if now_comp >= sched_dt:
                    cid = c.get("id")
                    org_id = c.get("org_id", "")
                    print(f"⏰ [Campaign Scheduler] Scheduled time reached for campaign '{c.get('name')}' ({cid}) at {sched_str}. Auto-launching...")
                    self.launch_campaign(org_id, cid)
                    triggered.append(cid)
            except Exception as e:
                print(f"⚠️ [Campaign Scheduler] Could not parse scheduled_date '{sched_str}' for campaign {c.get('id')}: {e}")

        return triggered

    def start_scheduler(self) -> None:
        """Starts background asyncio loop to periodically check and auto-launch scheduled campaigns."""
        if hasattr(self, "_scheduler_task") and self._scheduler_task and not self._scheduler_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        self._scheduler_task = loop.create_task(self._scheduler_loop())
        print("⏰ [Campaign Scheduler] Background campaign scheduler started (polling every 5s).")

    async def _scheduler_loop(self) -> None:
        """Background loop polling every 5s."""
        while True:
            try:
                await self.check_and_trigger_scheduled()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ [Campaign Scheduler] Loop error: {e}")
            await asyncio.sleep(5)
