import os
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
AUDIO_DIR = BASE_DIR / "audio"
CREDS_DIR = BASE_DIR / "credentials"
PRIVATE_DIR = BASE_DIR / "private"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CREDS_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables: check credentials/.env first, then root .env
creds_env = CREDS_DIR / ".env"
if creds_env.is_file():
    load_dotenv(creds_env, override=True)
load_dotenv(BASE_DIR / ".env")


def detect_ollama_host() -> str:
    """
    Auto-detects active Ollama port by probing ports 11434 and 11435 for live responses.
    """
    def is_alive(host_url: str) -> bool:
        try:
            url = f"{host_url.rstrip('/')}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                return resp.status == 200
        except Exception:
            return False

    env_host = os.getenv("OLLAMA_HOST")
    if env_host and is_alive(env_host):
        return env_host

    for port in [11434, 11435]:
        candidate = f"http://127.0.0.1:{port}"
        if is_alive(candidate):
            return candidate

    return env_host or "http://127.0.0.1:11434"



OLLAMA_HOST = detect_ollama_host()

# Available Local Models
AVAILABLE_MODELS = [
    "qwen2.5-coder:3b-instruct",
    "hf.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M",
]

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:3b-instruct")

# Voice Assistant System Prompt
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (
        "You are a helpful local voice assistant powered by Qwen 2.5, running completely offline on this computer.\n"
        "You are NOT ChatGPT and you are NOT made by OpenAI. Your model is Qwen.\n"
        "Keep your answers short and conversational (1 to 3 sentences maximum).\n"
        "Do not use markdown, bullet points, asterisks, or code blocks.\n"
        "Do not give unnecessary explanations.\n"
        "Speak naturally as if talking out loud."
    ),
)

# STT Configuration (faster-whisper)
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" or "cuda"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CACHE_DIR = MODELS_DIR / "whisper"

# TTS Configuration (Piper)
PIPER_DIR = MODELS_DIR / "piper"
PIPER_DIR.mkdir(parents=True, exist_ok=True)
PIPER_VOICE_NAME = os.getenv("PIPER_VOICE_NAME", "en_US-lessac-medium")
PIPER_MODEL_PATH = PIPER_DIR / f"{PIPER_VOICE_NAME}.onnx"
PIPER_CONFIG_PATH = PIPER_DIR / f"{PIPER_VOICE_NAME}.onnx.json"

# Audio Hardware Devices
ALSA_INPUT_DEVICE = os.getenv("ALSA_INPUT_DEVICE", "hw:3,0")
ALSA_OUTPUT_DEVICE = os.getenv("ALSA_OUTPUT_DEVICE", "plughw:2,0")
ALSA_DEVICE = ALSA_OUTPUT_DEVICE

# Target Processing Sample Rate
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# Silero VAD Settings
VAD_MODEL_PATH = MODELS_DIR / "silero_vad.onnx"
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.40"))
SILENCE_DURATION_THRESHOLD = float(os.getenv("SILENCE_DURATION_THRESHOLD", "1.0"))

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
PUBLIC_WEBHOOK_URL = os.getenv("PUBLIC_WEBHOOK_URL", "")  # e.g., https://xxxx.ngrok-free.app

# JWT Authentication Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "voicing-ai-production-hmac-sha256-jwt-secret-signing-key-32bytes-min-2026")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

