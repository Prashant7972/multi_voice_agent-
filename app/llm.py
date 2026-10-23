import json
import logging
import re
from typing import List, Dict, Generator, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import GEMINI_API_KEY, LLM_MODEL, SYSTEM_PROMPT, OLLAMA_HOST, LLM_TEMPERATURE

logger = logging.getLogger("voice_agent.llm")


def clean_text_for_speech(text: str) -> str:
    """
    Cleans markdown formatting and special symbols so the text sounds natural
    when read aloud by Text-To-Speech (TTS).
    """
    if not text:
        return ""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove bold / italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Remove headers (# Title)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove bullet points and list markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove emojis or unusual symbols if necessary
    text = re.sub(r"[\xa0\u200b\u200e\u200f]", " ", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text


class GeminiApiClient:
    """
    High-performance direct REST API client for Google Gemini (Generative Language API).
    Maintains persistent HTTP keep-alive connection pooling, 0-budget thinking mode for
    sub-second conversational voice latency, automatic multi-key rotation, and rapid failovers.
    """
    def __init__(self, api_key: str = GEMINI_API_KEY):
        raw_keys = api_key or GEMINI_API_KEY
        if isinstance(raw_keys, str):
            self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        elif isinstance(raw_keys, (list, tuple)):
            self.api_keys = [str(k).strip() for k in raw_keys if k and str(k).strip()]
        else:
            self.api_keys = [str(raw_keys)]
        if not self.api_keys:
            self.api_keys = [""]
        self.key_index = 0
        self.api_key = self.api_keys[self.key_index]
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.session = requests.Session()
        # Keep-alive connection pooling eliminates per-turn TLS/TCP handshake overhead
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=Retry(
                total=1,
                backoff_factor=0.2,
                status_forcelist=[502, 503, 504],
                raise_on_status=False,
            ),
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def rotate_key(self) -> str:
        """Rotates to next available API key in round-robin fashion."""
        if len(self.api_keys) > 1:
            self.key_index = (self.key_index + 1) % len(self.api_keys)
            self.api_key = self.api_keys[self.key_index]
            logger.info("Rotated to next Gemini API key (index %d/%d)", self.key_index + 1, len(self.api_keys))
        return self.api_key

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Any:
        options = options or {}
        payload: Dict[str, Any] = {}
        contents: List[Dict[str, Any]] = []
        sys_parts: List[Dict[str, str]] = []

        # Parse messages into Gemini schema: system_instruction + contents
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content:
                continue
            if role == "system":
                sys_parts.append({"text": content})
            else:
                gemini_role = "model" if role in ("assistant", "model") else "user"
                contents.append({"role": gemini_role, "parts": [{"text": content}]})

        if sys_parts:
            payload["system_instruction"] = {"parts": sys_parts}
        payload["contents"] = contents

        # Generation config with thinkingBudget: 0 for instant conversational voice response
        gen_config: Dict[str, Any] = {
            "temperature": float(options.get("temperature", 0.7)),
            "topP": float(options.get("top_p", 0.9)),
            "thinkingConfig": {"thinkingBudget": 0},
        }
        if "num_predict" in options and options["num_predict"]:
            gen_config["maxOutputTokens"] = int(options["num_predict"])
        else:
            gen_config["maxOutputTokens"] = 250
        payload["generationConfig"] = gen_config

        models_to_try = [model]
        candidates = [
            "gemini-3-flash-preview",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
        ]
        for c in candidates:
            if c not in models_to_try:
                models_to_try.append(c)

        # Attempt calls with key rotation and model failover
        num_keys = max(1, len(self.api_keys))
        last_error = None

        for key_attempt in range(num_keys):
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            }

            for i, target_model in enumerate(models_to_try):
                is_last_model = (i == len(models_to_try) - 1)
                is_last_overall = is_last_model and (key_attempt == num_keys - 1)
                next_model = models_to_try[i + 1] if not is_last_model else None

                try:
                    if stream:
                        url = f"{self.base_url}/{target_model}:streamGenerateContent?alt=sse"
                        r = self.session.post(url, headers=headers, json=payload, stream=True, timeout=(3.0, 8.0))
                        if r.status_code == 429 and len(self.api_keys) > 1 and key_attempt < num_keys - 1:
                            logger.warning("Gemini key hit quota on %s, rotating key...", target_model)
                            self.rotate_key()
                            headers["x-goog-api-key"] = self.api_key
                            break  # Retry from primary model with new key
                        if r.status_code in (429, 503, 404, 400) and not is_last_overall:
                            logger.warning("Gemini model %s returned status %d, instantly failing over to %s", target_model, r.status_code, next_model or "next key")
                            continue
                        if r.status_code != 200:
                            logger.error("Gemini stream error [%d]: %s", r.status_code, r.text)
                            r.raise_for_status()

                        def stream_generator():
                            for line in r.iter_lines(decode_unicode=True):
                                if line and line.startswith("data: "):
                                    raw_json = line[6:].strip()
                                    try:
                                        data = json.loads(raw_json)
                                        candidates_data = data.get("candidates", [])
                                        if candidates_data:
                                            parts = candidates_data[0].get("content", {}).get("parts", [])
                                            for p in parts:
                                                t = p.get("text", "")
                                                if t:
                                                    yield {"message": {"content": t}}
                                    except Exception as parse_err:
                                        logger.warning("Error parsing Gemini stream chunk: %s", parse_err)

                        return stream_generator()
                    else:
                        url = f"{self.base_url}/{target_model}:generateContent"
                        r = self.session.post(url, headers=headers, json=payload, timeout=(3.0, 8.0))
                        if r.status_code == 429 and len(self.api_keys) > 1 and key_attempt < num_keys - 1:
                            logger.warning("Gemini key hit quota on %s, rotating key...", target_model)
                            self.rotate_key()
                            headers["x-goog-api-key"] = self.api_key
                            break  # Retry from primary model with new key
                        if r.status_code in (429, 503, 404, 400) and not is_last_overall:
                            logger.warning("Gemini model %s returned status %d, instantly failing over to %s", target_model, r.status_code, next_model or "next key")
                            continue
                        if r.status_code != 200:
                            logger.error("Gemini generate error [%d]: %s", r.status_code, r.text)
                            r.raise_for_status()

                        data = r.json()
                        candidates_data = data.get("candidates", [])
                        reply_text = ""
                        if candidates_data:
                            parts = candidates_data[0].get("content", {}).get("parts", [])
                            reply_text = "".join(p.get("text", "") for p in parts if "text" in p)
                        return {"message": {"content": reply_text}}
                except Exception as e:
                    last_error = e
                    if not is_last_overall:
                        logger.warning("Gemini model %s request failed (%s), failing over...", target_model, e)
                        continue
                    # Check if local Ollama is running as ultimate emergency fallback
                    try:
                        import ollama
                        from app.config import OLLAMA_HOST
                        ollama_client = ollama.Client(host=OLLAMA_HOST)
                        installed_models = ollama_client.list().get("models", [])
                        if installed_models:
                            local_model = installed_models[0].get("name") or installed_models[0].get("model")
                            logger.info("All Gemini cloud options exhausted. Falling back to local Ollama model %s...", local_model)
                            if stream:
                                stream_res = ollama_client.chat(model=local_model, messages=messages, stream=True, options=options)
                                def ollama_stream_gen():
                                    for chk in stream_res:
                                        c = chk.get("message", {}).get("content", "")
                                        if c:
                                            yield {"message": {"content": c}}
                                return ollama_stream_gen()
                            else:
                                return ollama_client.chat(model=local_model, messages=messages, options=options)
                    except Exception:
                        pass
                    raise last_error


class LLMClient:
    """
    High-level LLM conversational client supporting both Local Ollama (unlimited offline)
    and Google Gemini Cloud API. Automatically routes to the selected model and provides
    instant failover between cloud and local engines.
    """
    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = LLM_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        host: str = OLLAMA_HOST,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or LLM_MODEL
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.host = host
        self.gemini_client = GeminiApiClient(api_key=self.api_key)
        self.client = self.gemini_client  # Backward compatibility
        import ollama
        self.ollama_client = ollama.Client(host=self.host)
        self.messages: List[Dict[str, str]] = []
        self.clear_history()

    def is_gemini(self) -> bool:
        """Determines if the active model is a Google Gemini cloud model."""
        return "gemini" in self.model.lower()

    def clear_history(self) -> None:
        """Reset conversation history with system instruction."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def set_model(self, model_name: str) -> None:
        """Dynamically switch model and adjust system prompt identity."""
        self.model = model_name
        if self.is_gemini():
            self.system_prompt = (
                f"You are a helpful voice assistant powered by Google Gemini ({model_name}).\n"
                "Keep your answers short and conversational (1 to 3 sentences maximum).\n"
                "Do not use markdown, bullet points, asterisks, or code blocks.\n"
                "Do not give unnecessary explanations.\n"
                "Speak naturally as if talking out loud."
            )
        else:
            self.system_prompt = (
                f"You are a helpful local voice assistant powered by {model_name}, running completely offline and unlimited on this computer.\n"
                "You are NOT ChatGPT and you are NOT made by OpenAI.\n"
                "Keep your answers short and conversational (1 to 3 sentences maximum).\n"
                "Do not use markdown, bullet points, asterisks, or code blocks.\n"
                "Do not give unnecessary explanations.\n"
                "Speak naturally as if talking out loud."
            )
        self.clear_history()

    def set_system_prompt(self, system_prompt: str) -> None:
        """Dynamically updates the active system prompt without wiping conversation context."""
        if system_prompt and system_prompt.strip():
            self.system_prompt = system_prompt.strip()
            if self.messages and self.messages[0].get("role") == "system":
                self.messages[0]["content"] = self.system_prompt
            else:
                self.messages.insert(0, {"role": "system", "content": self.system_prompt})

    def chat(
        self,
        user_text: str,
        max_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        stateless: bool = False,
    ) -> str:
        """
        Send user message with conversation context to Gemini or Ollama.
        Supports explicit session history to prevent cross-caller state leakage in concurrent requests.
        """
        if history is not None or stateless:
            effective_messages = []
            sys_content = system_prompt or self.system_prompt
            if sys_content:
                effective_messages.append({"role": "system", "content": sys_content})
            if history:
                for item in history:
                    if item.get("role") in ("user", "assistant") and item.get("content"):
                        effective_messages.append({"role": item["role"], "content": item["content"]})
            effective_messages.append({"role": "user", "content": user_text})
        else:
            self.messages.append({"role": "user", "content": user_text})
            effective_messages = self.messages

        options: Dict[str, Any] = {
            "temperature": LLM_TEMPERATURE,
            "top_p": 0.9,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        if self.is_gemini():
            response = self.gemini_client.chat(
                model=self.model,
                messages=effective_messages,
                options=options,
            )
        else:
            try:
                response = self.ollama_client.chat(
                    model=self.model,
                    messages=effective_messages,
                    options=options,
                )
            except Exception:
                from app.config import detect_ollama_host
                new_host = detect_ollama_host()
                if new_host != self.host:
                    self.host = new_host
                    import ollama
                    self.ollama_client = ollama.Client(host=self.host)
                    response = self.ollama_client.chat(
                        model=self.model,
                        messages=effective_messages,
                        options=options,
                    )
                else:
                    raise

        reply = response["message"]["content"].strip()
        cleaned_reply = clean_text_for_speech(reply)
        if history is None and not stateless:
            self.messages.append({"role": "assistant", "content": cleaned_reply})
        return cleaned_reply

    def stream_chat(self, user_text: str, max_tokens: Optional[int] = None) -> Generator[str, None, str]:
        """
        Stream response chunks from Gemini or Ollama.
        Yields raw chunks and returns the accumulated cleaned response upon completion.
        """
        self.messages.append({"role": "user", "content": user_text})

        options: Dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        if self.is_gemini():
            stream = self.gemini_client.chat(
                model=self.model,
                messages=self.messages,
                stream=True,
                options=options,
            )
        else:
            try:
                stream = self.ollama_client.chat(
                    model=self.model,
                    messages=self.messages,
                    stream=True,
                    options=options,
                )
            except Exception:
                from app.config import detect_ollama_host
                new_host = detect_ollama_host()
                if new_host != self.host:
                    self.host = new_host
                    import ollama
                    self.ollama_client = ollama.Client(host=self.host)
                    stream = self.ollama_client.chat(
                        model=self.model,
                        messages=self.messages,
                        stream=True,
                        options=options,
                    )
                else:
                    raise

        full_content = []
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                full_content.append(content)
                yield content

        raw_reply = "".join(full_content).strip()
        cleaned_reply = clean_text_for_speech(raw_reply)
        self.messages.append({"role": "assistant", "content": cleaned_reply})
        return cleaned_reply

    def stream_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, str]:
        """
        Streams token chunks for a prompt with optional prior conversation history window.
        """
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        if history:
            for item in history:
                role = item.get("role", "user")
                content = item.get("content", "")
                if role in ("user", "assistant") and content:
                    msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": prompt})

        options: Dict[str, Any] = {
            "temperature": LLM_TEMPERATURE,
            "top_p": 0.9,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        if self.is_gemini():
            stream = self.gemini_client.chat(
                model=self.model,
                messages=msgs,
                stream=True,
                options=options,
            )
        else:
            try:
                stream = self.ollama_client.chat(
                    model=self.model,
                    messages=msgs,
                    stream=True,
                    options=options,
                )
            except Exception:
                from app.config import detect_ollama_host
                new_host = detect_ollama_host()
                if new_host != self.host:
                    self.host = new_host
                    import ollama
                    self.ollama_client = ollama.Client(host=self.host)
                    stream = self.ollama_client.chat(
                        model=self.model,
                        messages=msgs,
                        stream=True,
                        options=options,
                    )
                else:
                    raise

        full_content = []
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                full_content.append(content)
                yield content

        return "".join(full_content)

    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        clean_speech: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a full multi-turn conversation messages array to Gemini or Ollama.
        Returns the assistant reply (optionally cleaned for speech).
        """
        effective_temp = LLM_TEMPERATURE if temperature is None else temperature
        options: Dict[str, Any] = {
            "temperature": effective_temp,
            "top_p": 0.9,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        if self.is_gemini():
            response = self.gemini_client.chat(
                model=self.model,
                messages=messages,
                options=options,
            )
        else:
            try:
                response = self.ollama_client.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                )
            except Exception:
                from app.config import detect_ollama_host
                new_host = detect_ollama_host()
                if new_host != self.host:
                    self.host = new_host
                    import ollama
                    self.ollama_client = ollama.Client(host=self.host)
                    response = self.ollama_client.chat(
                        model=self.model,
                        messages=messages,
                        options=options,
                    )
                else:
                    raise

        reply = response["message"]["content"].strip()
        if clean_speech:
            return clean_text_for_speech(reply)
        return reply

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.messages)
