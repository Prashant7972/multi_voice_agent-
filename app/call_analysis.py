"""
Post-Call Intelligence & Conversation Analytics Service
Provides automated and on-demand conversation analysis:
- Executive 2-3 sentence synopsis
- Sentiment categorization & normalized polarity score (-1.0 to 1.0)
- Lead qualification scoring (1-100) based on commercial intent & urgency
- Action items & follow-up task extraction
- Topic extraction & automated CRM tagging
- Local Ollama LLM structured JSON generation with deterministic heuristic fallback
"""
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger("voice_agent.analysis")


class CallAnalyzer:
    """
    Analyzes multi-turn dialogue transcripts to extract executive summaries,
    sentiment polarity, lead scores, action items, and CRM tags.
    """
    _instance: Optional["CallAnalyzer"] = None

    @classmethod
    def get_instance(cls) -> "CallAnalyzer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def format_dialogue(self, messages: List[Dict[str, Any]]) -> str:
        """Formats turn-by-turn messages into a clean speaker dialogue script."""
        lines = []
        for m in messages:
            role = m.get("role", "user")
            text = (m.get("text") or m.get("content") or "").strip()
            if not text:
                continue
            speaker = "Caller" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    def analyze_transcript(
        self,
        messages: List[Dict[str, Any]],
        agent_name: str = "Voice Assistant",
        llm_client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes a conversation transcript.
        Tries local LLM inference first, falling back cleanly to rule-based heuristics.
        """
        if not messages:
            return self._empty_analysis()

        dialogue = self.format_dialogue(messages)
        if not dialogue.strip():
            return self._empty_analysis()

        # Attempt structured analysis with local LLM if client is provided
        if llm_client:
            try:
                analysis = self._llm_analysis(dialogue, agent_name, llm_client)
                if analysis:
                    return analysis
            except Exception as e:
                logger.warning("LLM call analysis encountered error, using heuristic fallback: %s", e)

        # Fallback to deterministic heuristic analysis
        return self._heuristic_fallback(messages, agent_name)

    def _llm_analysis(
        self,
        dialogue: str,
        agent_name: str,
        llm_client: Any,
    ) -> Optional[Dict[str, Any]]:
        """Queries local Ollama LLM for structured JSON analysis."""
        system_instruction = (
            "You are an expert conversational intelligence and CRM analyst. "
            "Analyze the provided telephone/chat dialogue and output ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "summary": "Concise 2-3 sentence executive synopsis of the call",\n'
            '  "sentiment": "Positive" | "Neutral" | "Negative",\n'
            '  "sentiment_score": 0.0 to 1.0,\n'
            '  "lead_score": 1 to 100,\n'
            '  "intent": "Primary caller purpose (e.g. Enterprise Sales, Support, Account Billing)",\n'
            '  "customer_satisfaction": "High" | "Medium" | "Low",\n'
            '  "key_topics": ["Topic 1", "Topic 2"],\n'
            '  "action_items": ["Action item 1", "Action item 2"],\n'
            '  "crm_tags": ["tag1", "tag2"]\n'
            "}\n"
            "Do NOT wrap in markdown formatting. Return raw JSON only."
        )

        user_prompt = f"Dialogue Transcript with AI Agent ({agent_name}):\n\n{dialogue}\n\nAnalyze now:"

        msgs = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ]

        # Call chat_messages with clean_speech=False to receive raw JSON text
        if hasattr(llm_client, "chat_messages"):
            raw_output = llm_client.chat_messages(msgs, max_tokens=400, clean_speech=False)
        elif hasattr(llm_client, "chat"):
            raw_output = llm_client.chat(f"{system_instruction}\n\n{user_prompt}", max_tokens=400)
        elif hasattr(llm_client, "generate"):
            raw_output = llm_client.generate(f"{system_instruction}\n\n{user_prompt}", max_tokens=400)
        else:
            raw_output = str(llm_client(user_prompt))

        return self._parse_json_response(raw_output)

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts and validates JSON from model output."""
        if not text:
            return None

        # Clean markdown code fences if present
        clean = text.strip()
        if "```json" in clean:
            clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in clean:
            clean = clean.split("```", 1)[1].split("```", 1)[0].strip()

        # Find first '{' and last '}'
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean = clean[start : end + 1]

        try:
            data = json.loads(clean)
            if not isinstance(data, dict):
                return None

            # Validate and normalize fields
            summary = str(data.get("summary") or "").strip()
            if not summary:
                return None

            raw_sentiment = str(data.get("sentiment", "neutral")).lower()
            sentiment = raw_sentiment if raw_sentiment in ("positive", "neutral", "negative") else "neutral"

            try:
                sentiment_score = float(data.get("sentiment_score", 0.5))
                sentiment_score = max(0.0, min(1.0, sentiment_score))
            except (ValueError, TypeError):
                sentiment_score = 0.5

            try:
                lead_score = int(data.get("lead_score", 50))
                lead_score = max(1, min(100, lead_score))
            except (ValueError, TypeError):
                lead_score = 50

            intent = str(data.get("intent") or "General Inquiry").strip()
            sat = str(data.get("customer_satisfaction", "Medium")).capitalize()
            satisfaction = sat if sat in ("High", "Medium", "Low") else "Medium"

            key_topics = [str(t).strip() for t in data.get("key_topics", []) if str(t).strip()]
            action_items = [str(a).strip() for a in data.get("action_items", []) if str(a).strip()]
            crm_tags = [re.sub(r"[^a-zA-Z0-9_-]", "", str(tag).strip().lower()) for tag in data.get("crm_tags", []) if str(tag).strip()]

            return {
                "summary": summary,
                "sentiment": sentiment,
                "sentiment_score": round(sentiment_score, 2),
                "lead_score": lead_score,
                "lead_grade": self._grade_lead(lead_score),
                "intent": intent,
                "customer_satisfaction": satisfaction,
                "key_topics": key_topics[:6],
                "action_items": action_items[:5],
                "crm_tags": crm_tags[:8],
                "analyzed_at": time.time(),
                "engine": "ollama_llm",
            }
        except Exception as e:
            logger.debug("Failed to parse LLM JSON analysis: %s", e)
            return None

    def _heuristic_fallback(
        self,
        messages: List[Dict[str, Any]],
        agent_name: str,
    ) -> Dict[str, Any]:
        """
        Fast deterministic heuristic extractor that guarantees reliable analysis
        even when local LLM is offline or high latency.
        """
        caller_texts = []
        assistant_texts = []
        full_text = []

        for m in messages:
            role = m.get("role", "user")
            t = (m.get("text") or m.get("content") or "").strip()
            if not t:
                continue
            full_text.append(t)
            if role == "user":
                caller_texts.append(t)
            else:
                assistant_texts.append(t)

        combined_caller = " ".join(caller_texts).lower()
        combined_all = " ".join(full_text).lower()

        # 1. Sentiment Lexicon Scoring
        pos_words = {
            "thank", "thanks", "great", "helpful", "perfect", "good", "love",
            "awesome", "excellent", "appreciate", "resolved", "clear", "happy",
            "wonderful", "fantastic", "glad", "satisfied", "yes", "sure"
        }
        neg_words = {
            "issue", "problem", "frustrated", "bad", "terrible", "cancel",
            "fail", "failed", "slow", "error", "broken", "unhappy", "angry",
            "confused", "refund", "waste", "worst", "bug", "wrong", "no"
        }

        pos_count = sum(1 for word in re.findall(r"\b\w+\b", combined_all) if word in pos_words)
        neg_count = sum(1 for word in re.findall(r"\b\w+\b", combined_all) if word in neg_words)

        net_polarity = (pos_count - neg_count) / max(1, pos_count + neg_count)
        # Normalize to 0.0 - 1.0 range
        normalized_score = round(max(0.05, min(0.95, (net_polarity + 1.0) / 2.0)), 2)

        if net_polarity >= 0.15:
            sentiment = "positive"
            satisfaction = "High"
        elif net_polarity <= -0.15:
            sentiment = "negative"
            satisfaction = "Low"
        else:
            sentiment = "neutral"
            satisfaction = "Medium"

        # 2. Lead Scoring (Intent, Urgency, Commercial Signals)
        lead_score = 40  # baseline
        high_commercial_signals = {
            "pricing", "price", "cost", "quote", "budget", "purchase", "buy",
            "contract", "enterprise", "subscription", "schedule", "demo", "sales",
            "license", "order", "seat", "seats", "hire", "consultant"
        }
        urgency_signals = {"urgent", "asap", "immediately", "today", "now", "soon", "critical"}
        churn_signals = {"cancel", "refund", "close account", "stop", "unsubscribe"}

        commercial_hits = sum(1 for word in high_commercial_signals if word in combined_caller)
        urgency_hits = sum(1 for word in urgency_signals if word in combined_caller)
        churn_hits = sum(1 for word in churn_signals if word in combined_caller)

        lead_score += commercial_hits * 15
        lead_score += urgency_hits * 10
        lead_score -= churn_hits * 25
        lead_score += min(15, len(messages) * 3)  # engagement depth bonus
        lead_score = max(10, min(98, lead_score))

        # 3. Intent Identification
        if commercial_hits >= 2 or "demo" in combined_caller or "pricing" in combined_caller:
            intent = "Commercial Sales & Demo Request"
        elif "statement" in combined_caller or "billing" in combined_caller or "account" in combined_caller:
            intent = "Account & Billing Inquiry"
        elif "error" in combined_caller or "bug" in combined_caller or "help" in combined_caller:
            intent = "Technical Support & Troubleshooting"
        elif "transfer" in combined_caller or "agent" in combined_caller:
            intent = "Live Agent Transfer Request"
        else:
            intent = "General Voice Inquiry"

        # 4. Action Items Extraction
        action_items = []
        action_patterns = [
            r"(?:i will|let me|we will|i'll)\s+([^.!?\n]+[.!?]?)",
            r"(?:please\s+send|send\s+me|forward\s+to)\s+([^.!?\n]+[.!?]?)",
            r"(?:schedule|follow up|reach out)\s+([^.!?\n]+[.!?]?)",
        ]
        for pat in action_patterns:
            for text_turn in full_text:
                matches = re.findall(pat, text_turn, re.IGNORECASE)
                for m in matches:
                    cleaned_act = m.strip().rstrip(".!?")
                    if len(cleaned_act) > 8 and len(cleaned_act) < 80:
                        formatted = cleaned_act[0].upper() + cleaned_act[1:]
                        if formatted not in action_items:
                            action_items.append(formatted)

        if not action_items:
            if commercial_hits > 0:
                action_items.append("Follow up with caller regarding pricing and custom architecture review.")
            elif churn_hits > 0:
                action_items.append("Escalate account concerns to retention specialist.")
            else:
                action_items.append("Log call resolution and verify customer satisfaction status.")

        # 5. Key Topics
        key_topics = []
        topic_keywords = {
            "Enterprise Architecture": ["enterprise", "scale", "infrastructure"],
            "Pricing & Plans": ["pricing", "cost", "plan", "subscription", "quote"],
            "Account Statement": ["statement", "portfolio", "allocation", "balance"],
            "Support Assistance": ["support", "problem", "issue", "troubleshoot"],
            "Product Demo": ["demo", "walkthrough", "test", "show"],
            "Voice AI Setup": ["voice", "assistant", "agent", "microphone"],
        }
        for topic, kws in topic_keywords.items():
            if any(kw in combined_all for kw in kws):
                key_topics.append(topic)
        if not key_topics:
            key_topics = ["General Consultation", "Voice Assistant Interaction"]

        # 6. CRM Tags
        crm_tags = []
        if lead_score >= 75:
            crm_tags.append("hot-lead")
        elif lead_score >= 50:
            crm_tags.append("warm-lead")
        else:
            crm_tags.append("cold-inquiry")

        if commercial_hits > 0:
            crm_tags.append("commercial-interest")
        if churn_hits > 0:
            crm_tags.append("churn-risk")
        if any("transfer" in t.lower() for t in full_text):
            crm_tags.append("agent-transferred")
        crm_tags.append("voice-ai")

        # 7. Executive Summary
        first_user_turn = caller_texts[0] if caller_texts else "Customer called with an inquiry."
        last_asst_turn = assistant_texts[-1] if assistant_texts else f"{agent_name} handled the request."
        summary = (
            f"Caller contacted {agent_name} regarding: \"{first_user_turn[:90]}\". "
            f"The conversation concluded with: \"{last_asst_turn[:90]}\". "
            f"Overall sentiment was assessed as {sentiment.lower()} with a lead qualification score of {lead_score}/100."
        )

        return {
            "summary": summary,
            "sentiment": sentiment,
            "sentiment_score": normalized_score,
            "lead_score": lead_score,
            "lead_grade": self._grade_lead(lead_score),
            "intent": intent,
            "customer_satisfaction": satisfaction,
            "key_topics": key_topics[:6],
            "action_items": action_items[:5],
            "crm_tags": crm_tags[:8],
            "analyzed_at": time.time(),
            "engine": "heuristic_rule_engine",
        }

    def _grade_lead(self, score: int) -> str:
        """Returns standard lead grade: HOT, WARM, or COLD."""
        if score >= 75:
            return "HOT"
        elif score >= 45:
            return "WARM"
        return "COLD"

    def _empty_analysis(self) -> Dict[str, Any]:
        """Returns baseline analysis for empty sessions."""
        return {
            "summary": "No speech or messages recorded in this conversation session.",
            "sentiment": "neutral",
            "sentiment_score": 0.50,
            "lead_score": 50,
            "lead_grade": "WARM",
            "intent": "Unknown / Dropped Call",
            "customer_satisfaction": "Medium",
            "key_topics": [],
            "action_items": [],
            "crm_tags": ["dropped-call"],
            "analyzed_at": time.time(),
            "engine": "empty_session_fallback",
        }
