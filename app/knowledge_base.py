import json
import math
import os
import re
import subprocess
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

KNOWLEDGE_FILE = Path(__file__).resolve().parent / "data" / "knowledge_base.json"


def slugify(text: str) -> str:
    """Converts a title or name into a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-") or "doc"


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from PDF bytes using pdftotext or pure-Python text stream fallback."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        tmp_pdf.write(file_bytes)
        tmp_pdf_path = tmp_pdf.name

    extracted = ""
    try:
        if os.path.exists("/usr/bin/pdftotext"):
            txt_output = subprocess.run(
                ["/usr/bin/pdftotext", "-layout", tmp_pdf_path, "-"],
                capture_output=True,
                text=True,
                check=False,
            )
            if txt_output.returncode == 0 and txt_output.stdout.strip():
                extracted = txt_output.stdout.strip()
    except Exception as e:
        print(f"⚠️ [KnowledgeBase] pdftotext error: {e}")

    # Fallback basic text extraction if pdftotext didn't extract or wasn't available
    if not extracted:
        try:
            content = file_bytes.decode("latin1", errors="ignore")
            matches = re.findall(r"\(([^()]+)\)\s*Tj", content)
            if matches:
                extracted = " ".join(matches)
            else:
                matches_bt = re.findall(r"BT[\s\S]*?ET", content)
                extracted = " ".join(re.findall(r"[A-Za-z0-9 .,!?:;'\"]{3,}", " ".join(matches_bt)))
        except Exception as e:
            print(f"⚠️ [KnowledgeBase] PDF stream fallback error: {e}")

    try:
        os.remove(tmp_pdf_path)
    except Exception:
        pass

    return extracted.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extracts plain text from DOCX bytes using standard zipfile and xml parser."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_docx:
        tmp_docx.write(file_bytes)
        tmp_docx_path = tmp_docx.name

    text_parts = []
    try:
        with zipfile.ZipFile(tmp_docx_path, "r") as z:
            if "word/document.xml" in z.namelist():
                xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                for node in tree.iter():
                    if node.tag.endswith("}t") and node.text:
                        text_parts.append(node.text)
                    elif node.tag.endswith("}p"):
                        text_parts.append("\n")
    except Exception as e:
        print(f"⚠️ [KnowledgeBase] DOCX extraction error: {e}")
    finally:
        try:
            os.remove(tmp_docx_path)
        except Exception:
            pass

    return "".join(text_parts).strip()


def chunk_text(text: str, chunk_size_words: int = 250, overlap_words: int = 40) -> List[Dict[str, Any]]:
    """Splits plain text into overlapping semantic chunks for retrieval."""
    if not text or not text.strip():
        return []

    words = re.findall(r"\S+", text)
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size_words - overlap_words)
    chunk_index = 0

    for i in range(0, len(words), step):
        chunk_slice = words[i : i + chunk_size_words]
        chunk_str = " ".join(chunk_slice)
        if len(chunk_str.strip()) > 15:
            chunks.append({
                "chunk_id": f"chunk-{chunk_index + 1}",
                "text": chunk_str,
                "word_count": len(chunk_slice),
                "start_word_index": i,
            })
            chunk_index += 1
        if i + chunk_size_words >= len(words):
            break

    return chunks


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

CONVERSATIONAL_GREETINGS = {
    "hi", "hello", "hey", "hola", "howdy", "sup", "yo", "greetings",
    "hi there", "hello there", "hey there",
    "good morning", "good afternoon", "good evening", "good night", "good day",
    "how are you", "how are you doing", "how do you do", "hows it going", "how is it going", "how are things",
    "what's up", "whats up",
    "who are you", "what is your name", "what are you", "tell me about yourself",
    "what can you do", "help", "help me",
    "thank you", "thanks", "thank you very much", "thanks a lot", "thanks so much", "thx", "ty", "appreciate it",
    "bye", "goodbye", "see you", "see you later", "take care", "see ya", "have a good day", "have a nice day",
    "ok", "okay", "cool", "great", "nice", "awesome", "got it", "understood",
    "who made you", "what do you do", "who created you", "are you an ai", "are you a bot", "are you real"
}


def is_conversational_query(text: str) -> bool:
    """Checks if text is a common conversational greeting or chit-chat."""
    cleaned = re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
    if not cleaned:
        return True
    if cleaned in CONVERSATIONAL_GREETINGS:
        return True
    words = cleaned.split()
    if len(words) <= 3 and all(w in {"hi", "hello", "hey", "there", "assistant", "bot", "ai", "nova", "good", "morning", "afternoon", "evening", "day", "howdy", "yo", "sup"} for w in words):
        return True
    return False


def tokenize(text: str, filter_stopwords: bool = False) -> List[str]:
    """Tokenizes text into normalized lowercase alphanumeric words, optionally filtering stopwords."""
    words = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1]
    if filter_stopwords:
        return [w for w in words if w not in STOPWORDS]
    return words



class KnowledgeBaseManager:
    """
    Manages organization-scoped knowledge sources, multi-document ingestion,
    and BM25/TF-IDF similarity retrieval for Agent RAG.
    """
    _instance: Optional["KnowledgeBaseManager"] = None

    def __init__(self, storage_path: Path = KNOWLEDGE_FILE):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.sources: List[Dict[str, Any]] = []
        self.load()

    @classmethod
    def get_instance(cls) -> "KnowledgeBaseManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Loads knowledge base entries from JSON file, seeding defaults if absent."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.sources = data.get("sources", [])
                    # Backfill agent_id for legacy sources
                    for s in self.sources:
                        if not s.get("agent_id"):
                            s["agent_id"] = "general-voice-assistant"
            except Exception as e:
                print(f"⚠️ [KnowledgeBaseManager] Error reading {self.storage_path}: {e}")
                self.sources = []

        if not self.sources:
            # Seed default knowledge source for the primary organization
            default_content = (
                "Apex Voice AI Enterprise Protocol and Knowledge Base:\n\n"
                "1. Overview: Apex Voice AI provides low-latency local voice agents, telephony bridges via Twilio, "
                "and offline neural voice activity detection (Silero VAD).\n"
                "2. Model Architecture: Reasoning is handled by Qwen 2.5 Coder 3B and NVIDIA Nemotron 3 Nano offline LLMs. "
                "Spoken generation uses Piper neural TTS and transcription uses faster-whisper.\n"
                "3. Organization Policy: Access to administrative settings and knowledge base configuration is restricted "
                "to Organization Admins and Owners. Analysts have read-only analytics access.\n"
                "4. Customer Support SLAs: Critical enterprise telephony tickets have a 15-minute response SLA. "
                "Standard voice bot inquiries are resolved within 24 hours."
            )
            self.sources = [
                {
                    "id": "kb-apex-enterprise-docs",
                    "org_id": "apex-ai-innovations",
                    "agent_id": "general-voice-assistant",
                    "title": "Apex AI Enterprise Operations Manual",
                    "source_type": "document",
                    "filename": "apex_enterprise_manual.md",
                    "category": "Documentation",
                    "content": default_content,
                    "chunks": chunk_text(default_content),
                    "created_at": time.time() - 3600,
                    "updated_at": time.time() - 3600,
                    "added_by": "admin@apexai.io",
                    "word_count": len(re.findall(r"\S+", default_content)),
                    "file_size": len(default_content.encode("utf-8")),
                }
            ]
            self.save()

    def save(self) -> None:
        """Persists knowledge base entries to JSON storage."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"sources": self.sources}, f, indent=2)
        except Exception as e:
            print(f"⚠️ [KnowledgeBaseManager] Error saving to {self.storage_path}: {e}")

    def list_sources(self, org_id: str, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns knowledge sources belonging to org_id, optionally filtered to a specific agent_id."""
        if not org_id:
            return []
        org_id_clean = org_id.strip().lower()
        agent_id_clean = agent_id.strip().lower() if (agent_id and agent_id.strip()) else None

        res = []
        for s in self.sources:
            if s.get("org_id", "").strip().lower() != org_id_clean:
                continue
            if agent_id_clean and s.get("agent_id", "").strip().lower() != agent_id_clean:
                continue
            res.append(s)
        res.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return res

    def get_source(self, org_id: str, source_id: str, agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Finds a specific knowledge source within org_id, optionally validating agent_id."""
        org_id_clean = org_id.strip().lower()
        agent_id_clean = agent_id.strip().lower() if (agent_id and agent_id.strip()) else None
        for s in self.sources:
            if s.get("id") == source_id and s.get("org_id", "").strip().lower() == org_id_clean:
                if agent_id_clean and s.get("agent_id", "").strip().lower() != agent_id_clean:
                    continue
                return s
        return None

    def add_text_source(
        self,
        org_id: str,
        title: str,
        content: str,
        agent_id: Optional[str] = None,
        source_type: str = "text",
        category: str = "General",
        added_by: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingests a typed or manual text knowledge source into org_id for a specific agent."""
        if not content or not content.strip():
            raise ValueError("Knowledge content cannot be empty.")
        if not title or not title.strip():
            title = f"Knowledge Note - {time.strftime('%Y-%m-%d %H:%M')}"

        effective_agent_id = agent_id.strip() if (agent_id and agent_id.strip()) else "general-voice-assistant"
        source_id = f"kb-{int(time.time() * 1000)}-{slugify(title)[:20]}"
        chunks = chunk_text(content)
        words = re.findall(r"\S+", content)

        new_source = {
            "id": source_id,
            "org_id": org_id.strip().lower(),
            "agent_id": effective_agent_id,
            "title": title.strip(),
            "source_type": source_type,
            "category": category,
            "content": content.strip(),
            "chunks": chunks,
            "created_at": time.time(),
            "updated_at": time.time(),
            "added_by": added_by.strip() or "system",
            "word_count": len(words),
            "file_size": len(content.encode("utf-8")),
            "metadata": metadata or {},
        }

        self.sources.append(new_source)
        self.save()
        print(f"📚 [Knowledge Base] Ingested {source_type} source '{title}' for agent '{effective_agent_id}' in org '{org_id}' ({len(chunks)} chunks)")
        return new_source

    def add_document_source(
        self,
        org_id: str,
        filename: str,
        file_bytes: bytes,
        agent_id: Optional[str] = None,
        category: str = "Document",
        added_by: str = "",
    ) -> Dict[str, Any]:
        """Parses and ingests a document (.pdf, .docx, .txt, .md, .csv, .json) into org_id for an agent."""
        ext = Path(filename).suffix.lower()
        extracted_text = ""

        if ext == ".pdf":
            extracted_text = extract_text_from_pdf(file_bytes)
        elif ext in [".docx", ".doc"]:
            extracted_text = extract_text_from_docx(file_bytes)
        elif ext in [".txt", ".md", ".csv", ".json", ".xml", ".html", ".log"]:
            try:
                extracted_text = file_bytes.decode("utf-8", errors="replace")
            except Exception:
                extracted_text = file_bytes.decode("latin1", errors="replace")
        else:
            try:
                extracted_text = file_bytes.decode("utf-8", errors="replace")
            except Exception:
                pass

        if not extracted_text or not extracted_text.strip():
            raise ValueError(f"Could not extract readable text from '{filename}'.")

        title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        metadata = {
            "original_filename": filename,
            "file_extension": ext,
            "raw_byte_size": len(file_bytes),
        }

        return self.add_text_source(
            org_id=org_id,
            title=title,
            content=extracted_text,
            agent_id=agent_id,
            source_type="document",
            category=category,
            added_by=added_by,
            metadata=metadata,
        )

    def add_voice_source(
        self,
        org_id: str,
        title: str,
        audio_path_or_bytes: Any,
        stt_client: Any,
        agent_id: Optional[str] = None,
        added_by: str = "",
        category: str = "Voice Memo",
    ) -> Dict[str, Any]:
        """Transcribes spoken audio and ingests as a voice knowledge source for an agent."""
        temp_audio_file = None
        if isinstance(audio_path_or_bytes, bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_path_or_bytes)
                temp_audio_file = f.name
            audio_target = temp_audio_file
        else:
            audio_target = str(audio_path_or_bytes)

        try:
            transcribed_text = stt_client.transcribe(audio_target)
        except Exception as e:
            print(f"⚠️ [KnowledgeBase] STT transcription failed: {e}")
            raise ValueError(f"Voice transcription failed: {e}")
        finally:
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    os.remove(temp_audio_file)
                except Exception:
                    pass

        if not transcribed_text or not transcribed_text.strip():
            raise ValueError("No speech or audio words were recognized in the recording.")

        if not title or not title.strip():
            title = f"Voice Knowledge - {time.strftime('%b %d, %H:%M')}"

        return self.add_text_source(
            org_id=org_id,
            title=title,
            content=transcribed_text,
            agent_id=agent_id,
            source_type="voice",
            category=category,
            added_by=added_by,
            metadata={"transcription_engine": "Whisper"},
        )

    def delete_source(self, org_id: str, source_id: str) -> bool:
        """Deletes a knowledge source strictly within org_id."""
        org_id_clean = org_id.strip().lower()
        initial_len = len(self.sources)
        self.sources = [
            s for s in self.sources
            if not (s.get("id") == source_id and s.get("org_id", "").strip().lower() == org_id_clean)
        ]
        if len(self.sources) < initial_len:
            self.save()
            print(f"🗑️ [Knowledge Base] Deleted source '{source_id}' from org '{org_id}'")
            return True
        return False

    def search(self, org_id: str, query: str, agent_id: Optional[str] = None, top_k: int = 4, min_score: float = 1.0) -> List[Dict[str, Any]]:
        """
        Performs TF-IDF / BM25 style keyword and semantic retrieval
        STRICTLY SCOPED to the provided org_id and optional agent_id.
        """
        if not org_id or not query or not query.strip():
            return []

        # If query is purely conversational greeting or chit-chat, skip RAG retrieval
        if is_conversational_query(query):
            return []

        org_sources = self.list_sources(org_id=org_id, agent_id=agent_id)
        if not org_sources:
            return []

        # Filter query tokens using stopwords so stopwords don't generate false hits
        query_tokens = tokenize(query, filter_stopwords=True)
        if not query_tokens:
            return []

        all_chunks = []
        for s in org_sources:
            for c in s.get("chunks", []):
                all_chunks.append({
                    "source_id": s.get("id"),
                    "source_title": s.get("title"),
                    "source_type": s.get("source_type"),
                    "category": s.get("category"),
                    "agent_id": s.get("agent_id"),
                    "chunk_id": c.get("chunk_id"),
                    "text": c.get("text"),
                    "tokens": tokenize(c.get("text", "")),
                })

        if not all_chunks:
            return []

        N = len(all_chunks)
        doc_freq = {}
        for token in set(query_tokens):
            df = sum(1 for c in all_chunks if token in c["tokens"])
            doc_freq[token] = df

        scored_chunks = []
        query_text_lower = query.lower().strip()

        for c in all_chunks:
            tokens = c["tokens"]
            if not tokens:
                continue

            score = 0.0
            matched_terms = 0
            for q_tok in query_tokens:
                tf = tokens.count(q_tok)
                stem_matches = [
                    t for t in tokens
                    if len(q_tok) >= 4 and len(t) >= 4 and (t.startswith(q_tok) or q_tok.startswith(t)) and t != q_tok
                ]
                effective_tf = tf + len(stem_matches) * 0.75
                if effective_tf > 0:
                    matched_terms += 1
                    df = doc_freq.get(q_tok, 1)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    tf_norm = (effective_tf * 2.2) / (effective_tf + 1.2 * (0.25 + 0.75 * len(tokens) / 50.0))
                    score += tf_norm * max(0.8, idf)

            if query_text_lower in c["text"].lower() and len(query_text_lower) > 3:
                score += 4.0

            # Only return chunks with genuine keyword matches and score >= min_score
            if matched_terms > 0 and score >= min_score:
                scored_chunks.append({
                    "source_id": c["source_id"],
                    "source_title": c["source_title"],
                    "source_type": c["source_type"],
                    "category": c["category"],
                    "agent_id": c.get("agent_id"),
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "score": round(score, 3),
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def build_rag_context(self, org_id: str, query: str, agent_id: Optional[str] = None, top_k: int = 3, min_score: float = 1.0) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Builds a formatted prompt context block from retrieved knowledge chunks strictly scoped to agent_id.
        Returns (context_prompt_string, citations_list).
        If query is conversational or no relevant chunks match, returns ("", []).
        """
        if is_conversational_query(query):
            return "", []

        matches = self.search(org_id=org_id, query=query, agent_id=agent_id, top_k=top_k, min_score=min_score)
        if not matches:
            return "", []

        agent_label = f" (Agent: {agent_id})" if agent_id else ""
        context_lines = [
            f"\n[CONFIDENTIAL AGENT KNOWLEDGE BASE CONTEXT{agent_label}]:",
            "Use the following verified confidential knowledge if directly relevant to the user's inquiry:",
        ]
        citations = []

        for idx, m in enumerate(matches):
            context_lines.append(f"Source [{idx + 1}] ({m['source_title']}): {m['text']}")
            citations.append({
                "source_id": m["source_id"],
                "source_title": m["source_title"],
                "source_type": m["source_type"],
                "agent_id": m.get("agent_id"),
                "chunk_id": m["chunk_id"],
                "score": m["score"],
                "snippet": m["text"][:160] + "..." if len(m["text"]) > 160 else m["text"],
            })

        context_lines.append("[END KNOWLEDGE BASE CONTEXT]\n")
        return "\n".join(context_lines), citations

    def test_source_retrieval(
        self,
        org_id: str,
        source_id: str,
        agent_id: str,
        test_query: str,
    ) -> Dict[str, Any]:
        """
        Tests whether a specific knowledge source is present in and accessible to the given agent_id,
        and evaluates BM25 chunk retrieval against test_query.
        """
        org_id_clean = org_id.strip().lower()
        source = None
        for s in self.sources:
            if s.get("id") == source_id and s.get("org_id", "").strip().lower() == org_id_clean:
                source = s
                break

        if not source:
            return {
                "source_id": source_id,
                "found": False,
                "present_in_agent": False,
                "is_accessible": False,
                "message": f"Knowledge source '{source_id}' does not exist in organization '{org_id}'.",
                "matched_chunks": [],
            }

        source_agent_id = source.get("agent_id", "general-voice-assistant")
        target_agent_id = agent_id.strip() if agent_id else "general-voice-assistant"

        # Check confidential isolation: Is the source accessible to target_agent_id?
        is_accessible = (source_agent_id == target_agent_id or source_agent_id in ["all", "*"])

        if not is_accessible:
            return {
                "source_id": source_id,
                "source_title": source.get("title"),
                "source_type": source.get("source_type"),
                "category": source.get("category"),
                "source_agent_id": source_agent_id,
                "target_agent_id": target_agent_id,
                "found": True,
                "present_in_agent": False,
                "is_accessible": False,
                "message": f"Isolation Enforced: This knowledge base is confidential and scoped to agent '{source_agent_id}'. It is not accessible to '{target_agent_id}'.",
                "matched_chunks": [],
            }

        # Calculate chunk retrieval score against test_query
        query_tokens = tokenize(test_query, filter_stopwords=True) if test_query and test_query.strip() else []
        chunks = source.get("chunks", [])
        matched_chunks = []

        query_text_lower = test_query.lower().strip() if test_query else ""

        for c in chunks:
            chunk_text = c.get("text", "")
            tokens = tokenize(chunk_text)
            score = 0.0
            matched_terms = 0

            if query_tokens:
                for q_tok in query_tokens:
                    tf = tokens.count(q_tok)
                    if tf > 0:
                        matched_terms += 1
                        score += tf * 1.5

            if query_text_lower and query_text_lower in chunk_text.lower() and len(query_text_lower) > 3:
                score += 5.0

            if score > 0 or not query_tokens:
                matched_chunks.append({
                    "chunk_id": c.get("chunk_id"),
                    "text": chunk_text,
                    "score": round(score, 2),
                    "matched_terms": matched_terms,
                })

        matched_chunks.sort(key=lambda x: x["score"], reverse=True)
        has_matches = len(matched_chunks) > 0

        # If no chunks match, check if another document in this organization covers the query
        hint = ""
        if not has_matches and query_tokens:
            for s_other in self.sources:
                if s_other.get("id") != source_id and s_other.get("org_id", "").strip().lower() == org_id_clean:
                    for oc in s_other.get("chunks", []):
                        oc_text = oc.get("text", "").lower()
                        if any(q.lower() in oc_text for q in query_tokens if len(q) > 3):
                            hint = f"Topic matches '{s_other.get('title')}' (assigned to agent '{s_other.get('agent_id')}')."
                            break
                    if hint:
                        break

        if has_matches:
            msg = f"Knowledge source '{source.get('title')}' is present in agent '{target_agent_id}' and matched {len(matched_chunks)} chunk(s)."
        else:
            msg = f"Knowledge source '{source.get('title')}' is associated with agent '{target_agent_id}', but contains no content matching '{test_query}'."
            if hint:
                msg += f" ({hint})"

        return {
            "source_id": source_id,
            "source_title": source.get("title"),
            "source_type": source.get("source_type"),
            "category": source.get("category"),
            "source_agent_id": source_agent_id,
            "target_agent_id": target_agent_id,
            "found": True,
            "present_in_agent": True,
            "is_accessible": True,
            "has_matching_chunks": has_matches,
            "hint": hint,
            "total_chunks": len(chunks),
            "matched_chunks": matched_chunks[:5],
            "highest_score": matched_chunks[0]["score"] if matched_chunks else 0.0,
            "message": msg,
        }


def verify_knowledge_citations(
    reply_text: str,
    citations: List[Dict[str, Any]],
    query: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Ensures that citations are ONLY returned when the generated answer is actually
    informed by or gathered from the retrieved knowledge base.
    If the query was conversational, or if the answer states no info was found,
    or if the reply did not actually incorporate knowledge base content, citations are stripped.
    """
    if not citations or not reply_text:
        return reply_text, []

    if is_conversational_query(query):
        return reply_text.replace("[GROUNDED_IN_KNOWLEDGE_BASE]", "").strip(), []

    grounded_tag = "[GROUNDED_IN_KNOWLEDGE_BASE]"
    has_grounded_tag = grounded_tag in reply_text
    clean_reply = reply_text.replace(grounded_tag, "").strip()

    # Negative patterns indicating knowledge base wasn't used or had no answer
    reply_lower = clean_reply.lower()
    negative_patterns = [
        r"don't have (any )?information",
        r"do not have (any )?information",
        r"not (mentioned|found|available|present|contained) in (the|our) (knowledge|document|context|manual)",
        r"no information (is )?provided",
        r"cannot find (any )?information",
        r"not specified in",
        r"i couldn't find",
        r"i could not find",
        r"not in the knowledge base",
    ]
    for pattern in negative_patterns:
        if re.search(pattern, reply_lower):
            return clean_reply, []

    # If explicitly tagged as grounded and no negative statement, keep citations
    if has_grounded_tag:
        return clean_reply, citations

    # Otherwise verify lexical/factual grounding: did the reply actually use content from the citation?
    query_words = set(re.findall(r"\w+", query.lower()))
    verified_citations = []

    for c in citations:
        chunk_text = c.get("text", "") or c.get("snippet", "")
        # Extract informative words from the chunk (exclude common stopwords and words already in the user's query)
        chunk_words = [
            w for w in re.findall(r"\w+", chunk_text.lower())
            if len(w) >= 4 and w not in STOPWORDS and w not in query_words
        ]

        # Check how many distinct informative words from this chunk appear in the reply
        overlap_words = {w for w in chunk_words if w in reply_lower}

        # Check title mention
        title_words = [w for w in re.findall(r"\w+", c.get("source_title", "").lower()) if len(w) >= 4 and w not in STOPWORDS]
        title_in_reply = any(w in reply_lower for w in title_words)

        # Check for phrase overlap (3+ words in sequence)
        phrase_matched = False
        words_list = chunk_text.lower().split()
        for i in range(len(words_list) - 3):
            subphrase = " ".join(words_list[i:i+4])
            subphrase_clean = re.sub(r"[^\w\s]", "", subphrase)
            if len(subphrase_clean) > 15 and subphrase_clean in reply_lower:
                phrase_matched = True
                break

        if len(overlap_words) >= 2 or title_in_reply or phrase_matched:
            verified_citations.append(c)

    return clean_reply, verified_citations


