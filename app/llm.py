import re
from typing import List, Dict, Generator, Optional
import ollama
from app.config import OLLAMA_HOST, LLM_MODEL, SYSTEM_PROMPT


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


class LLMClient:
    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = LLM_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.host = host
        self.model = model
        self.system_prompt = system_prompt
        self.client = ollama.Client(host=self.host)
        self.messages: List[Dict[str, str]] = []
        self.clear_history()

    def clear_history(self) -> None:
        """Reset conversation history with system instruction."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def set_model(self, model_name: str) -> None:
        """Dynamically switch model and adjust system prompt identity."""
        self.model = model_name
        if "nemotron" in model_name.lower():
            model_id = "NVIDIA Nemotron"
        else:
            model_id = "Qwen 2.5"

        self.system_prompt = (
            f"You are a helpful local voice assistant powered by {model_id}, running completely offline on this computer.\n"
            "You are NOT ChatGPT and you are NOT made by OpenAI.\n"
            "Keep your answers short and conversational (1 to 3 sentences maximum).\n"
            "Do not use markdown, bullet points, asterisks, or code blocks.\n"
            "Do not give unnecessary explanations.\n"
            "Speak naturally as if talking out loud."
        )
        self.clear_history()

    def set_system_prompt(self, system_prompt: str) -> None:
        """Dynamically updates the active system prompt without unexpectedly wiping conversation context."""
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
        Send user message to Ollama with conversation context.
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

        options = {
            "temperature": 0.7,
            "top_p": 0.9,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        try:
            response = self.client.chat(
                model=self.model,
                messages=effective_messages,
                options=options,
            )
        except Exception as e:
            from app.config import detect_ollama_host
            new_host = detect_ollama_host()
            if new_host != self.host:
                print(f"🔄 [LLMClient] Switching Ollama host from {self.host} to {new_host}...")
                self.host = new_host
                self.client = ollama.Client(host=self.host)
                response = self.client.chat(
                    model=self.model,
                    messages=effective_messages,
                    options=options,
                )
            else:
                raise e

        reply = response["message"]["content"].strip()
        cleaned_reply = clean_text_for_speech(reply)
        if history is None and not stateless:
            self.messages.append({"role": "assistant", "content": cleaned_reply})
        return cleaned_reply


    def stream_chat(self, user_text: str, max_tokens: Optional[int] = None) -> Generator[str, None, str]:
        """
        Stream response chunks from Ollama.
        Yields raw chunks and returns the accumulated cleaned response upon completion.
        """
        self.messages.append({"role": "user", "content": user_text})

        options = {
            "temperature": 0.7,
            "top_p": 0.9,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        stream = self.client.chat(
            model=self.model,
            messages=self.messages,
            stream=True,
            options=options,
        )

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

        options = {
            "temperature": 0.7,
            "top_p": 0.9,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        try:
            stream = self.client.chat(
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
                self.client = ollama.Client(host=self.host)
                stream = self.client.chat(
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
    ) -> str:
        """
        Send a full multi-turn conversation messages array to Ollama.
        Returns the assistant reply (optionally cleaned for speech).
        """
        options = {
            "temperature": 0.7,
            "top_p": 0.9,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options,
            )
        except Exception as e:
            from app.config import detect_ollama_host
            new_host = detect_ollama_host()
            if new_host != self.host:
                print(f"🔄 [LLMClient] Switching Ollama host from {self.host} to {new_host}...")
                self.host = new_host
                self.client = ollama.Client(host=self.host)
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                )
            else:
                raise e

        reply = response["message"]["content"].strip()
        if clean_speech:
            return clean_text_for_speech(reply)
        return reply

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.messages)
