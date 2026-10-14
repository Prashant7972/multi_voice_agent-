import os
import re
import json
import base64
import audioop
import asyncio
from typing import Optional, Dict, Any, List
import numpy as np

from app.config import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    PUBLIC_WEBHOOK_URL,
    SILENCE_DURATION_THRESHOLD,
)
from app.tunnel import TunnelManager
from app.pipecat_agent import PipecatNeuralVAD, VADState


def clean_ansi(text: str) -> str:
    """Strips terminal color codes from exception messages."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text).strip()


class TwilioService:
    """
    Handles Twilio REST API interactions:
    - Verifying configuration
    - Generating TwiML for inbound and outbound streaming calls
    - Triggering outbound phone calls to connect callers to the local AI voice agent
    """

    def __init__(
        self,
        account_sid: str = TWILIO_ACCOUNT_SID,
        auth_token: str = TWILIO_AUTH_TOKEN,
        phone_number: str = TWILIO_PHONE_NUMBER,
        public_url: str = PUBLIC_WEBHOOK_URL,
    ):
        self.account_sid = account_sid.strip() if account_sid else ""
        self.auth_token = auth_token.strip() if auth_token else ""
        self.phone_number = phone_number.strip() if phone_number else ""
        self.public_url = public_url.strip().rstrip("/") if public_url else ""

    def is_configured(self) -> bool:
        """Checks if valid Twilio credentials are provided."""
        return bool(
            self.account_sid
            and self.auth_token
            and not self.account_sid.startswith("ACxxx")
            and not self.auth_token.startswith("your_")
        )

    def get_public_url(self, host: Optional[str] = None) -> str:
        """
        Determines the public HTTPS base URL.
        1. Checks PUBLIC_WEBHOOK_URL in .env
        2. Checks if TunnelManager is running
        3. Starts TunnelManager to get a free live tunnel if needed
        4. Fallback to host header
        """
        if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("CI") == "true" or os.getenv("ENVIRONMENT") == "test":
            if host and "localhost" not in host and "127.0.0.1" not in host:
                return f"https://{host}"
            return self.public_url or "https://mock-tunnel.example.com"

        tunnel = TunnelManager.get_instance()
        if tunnel.public_url:
            return tunnel.public_url

        if self.public_url and "trycloudflare.com" not in self.public_url:
            return self.public_url

        # Auto-start free tunnel
        new_url = tunnel.start()
        if new_url:
            self.public_url = new_url
            return self.public_url

        if host and "localhost" not in host and "127.0.0.1" not in host:
            return f"https://{host}"

        port = os.getenv("PORT", "8080")
        return f"http://localhost:{port}"

    def get_stream_url(self, host: Optional[str] = None) -> str:
        """
        Determines the public WebSocket stream URL (wss://).
        """
        base = self.get_public_url(host)
        cleaned = base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{cleaned}/twiml/stream"

    def generate_twiml(self, host: Optional[str] = None, mode: str = "gather") -> str:
        """
        Generates TwiML XML.
        'gather': Continuous interactive telephone conversation with zero disconnects.
        'stream': Raw bidirectional WebSocket audio streaming.
        """
        public_base = self.get_public_url(host)

        if mode == "stream":
            stream_url = self.get_stream_url(host)
            return (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Response>\n'
                '    <Say voice="Polly.Joanna-Neural">Connecting you to your local AI voice agent. Please speak freely.</Say>\n'
                '    <Connect>\n'
                f'        <Stream url="{stream_url}" />\n'
                '    </Connect>\n'
                '</Response>'
            )

        # Default 'gather' mode: Continuous interactive phone conversation
        action_url = f"{public_base.rstrip('/')}/twiml/respond"
        incoming_url = f"{public_base.rstrip('/')}/twiml/incoming"
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Response>\n'
            f'    <Gather input="speech" action="{action_url}" method="POST" timeout="4" speechTimeout="auto" bargeIn="true">\n'
            '        <Say voice="Polly.Joanna-Neural">Hello! I am your local AI voice assistant. Speak anytime and I will answer.</Say>\n'
            '    </Gather>\n'
            f'    <Redirect method="POST">{incoming_url}</Redirect>\n'
            '</Response>'
        )

    def make_call(
        self,
        to_phone: str,
        host: Optional[str] = None,
        agent_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Initiates an outbound phone call using the Twilio REST API.
        Optionally binds the call to a designated AI agent_id and campaign_id.
        Note: Twilio trial accounts require the 'url' parameter and reject the 'twiml' parameter.
        """
        if not to_phone or not to_phone.strip():
            return {"status": "error", "message": "Destination phone number is required."}

        to_phone = to_phone.strip()

        if not self.is_configured():
            return {
                "status": "mock_mode",
                "message": (
                    "Twilio credentials not configured in .env. "
                    "Running in simulated mode. Update TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN to place live calls."
                ),
                "to": to_phone,
                "agent_id": agent_id,
                "campaign_id": campaign_id,
            }

        # Resolve public webhook URL with agent and campaign bindings
        public_base = self.get_public_url(host)
        query_params = []
        if agent_id:
            query_params.append(f"agent_id={agent_id.strip()}")
        if campaign_id:
            query_params.append(f"campaign_id={campaign_id.strip()}")
        qs = f"?{'&'.join(query_params)}" if query_params else ""
        webhook_url = f"{public_base.rstrip('/')}/twiml/incoming{qs}"
        status_callback_url = f"{public_base.rstrip('/')}/twiml/status-callback{qs}"

        try:
            from twilio.rest import Client
            client = Client(self.account_sid, self.auth_token)
            # Use 'url=' instead of 'twiml=' for trial account compatibility
            call = client.calls.create(
                to=to_phone,
                from_=self.phone_number,
                url=webhook_url,
                status_callback=status_callback_url,
                status_callback_event=["initiated", "ringing", "answered", "completed"],
                status_callback_method="POST",
            )
            return {
                "status": "success",
                "call_sid": call.sid,
                "to": to_phone,
                "agent_id": agent_id,
                "campaign_id": campaign_id,
                "webhook_url": webhook_url,
                "message": f"Calling {to_phone}...",
            }
        except Exception as e:
            cleaned_err = clean_ansi(str(e))
            return {
                "status": "error",
                "message": f"Twilio call failed: {cleaned_err}",
                "to": to_phone,
            }


class TwilioAudioTranscoder:
    """
    Handles bidirectional audio format conversion:
    - Twilio Inbound: 8,000 Hz, 8-bit mu-law -> 16,000 Hz, 16-bit linear PCM (Whisper input)
    - Twilio Outbound: 22,050 Hz, 16-bit linear PCM (Piper TTS) -> 8,000 Hz, 8-bit mu-law
    """

    @staticmethod
    def mulaw_to_pcm16k(mulaw_bytes: bytes) -> bytes:
        """Converts 8kHz 8-bit mulaw to 16kHz 16-bit linear PCM."""
        if not mulaw_bytes:
            return b""
        pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
        pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
        return pcm_16k

    @staticmethod
    def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
        """Converts 16-bit linear PCM bytes to a normalized float32 numpy array for faster-whisper."""
        if not pcm_bytes:
            return np.array([], dtype=np.float32)
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    @staticmethod
    def tts_pcm_to_mulaw8k(pcm_bytes: bytes, in_sample_rate: int = 22050) -> bytes:
        """Converts Piper TTS 16-bit linear PCM to 8kHz 8-bit mu-law for Twilio."""
        if not pcm_bytes:
            return b""
        pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, in_sample_rate, 8000, None)
        mulaw_data = audioop.lin2ulaw(pcm_8k, 2)
        return mulaw_data


class TwilioMediaStreamSession:
    """
    Manages a single Twilio bi-directional WebSocket media stream call.
    Receives incoming 8kHz mulaw frames, detects voice and silence pauses,
    transcribes speech with faster-whisper, generates LLM reply,
    synthesizes speech with Piper, and streams mulaw audio chunks back to the caller.
    """

    def __init__(
        self,
        websocket: Any,
        stt_client: Any,
        llm_client: Any,
        tts_client: Any,
        silence_threshold: float = SILENCE_DURATION_THRESHOLD,
    ):
        self.ws = websocket
        self.stt = stt_client
        self.llm = llm_client
        self.tts = tts_client
        self.silence_threshold = silence_threshold

        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None
        self.is_active = True
        self.is_agent_speaking = False
        self.interrupted = False

        self.pcm_buffer = bytearray()
        self.speech_detected = False
        self.consecutive_silence_sec = 0.0
        # Pipecat Neural VAD to eliminate line static & background noise false interruptions
        self.vad = PipecatNeuralVAD(sample_rate=16000, confidence=0.75, start_secs=0.20, stop_secs=0.80, min_volume=0.06)

    async def handle(self):
        """Main WebSocket loop for Twilio streaming."""
        try:
            while self.is_active:
                raw_msg = await self.ws.receive_text()
                msg = json.loads(raw_msg)
                event = msg.get("event")

                if event == "connected":
                    print("📞 [Twilio] WebSocket event: connected")

                elif event == "start":
                    start_data = msg.get("start", {})
                    self.stream_sid = start_data.get("streamSid") or msg.get("streamSid")
                    self.call_sid = start_data.get("callSid")
                    print(f"📞 [Twilio] Call started! StreamSid={self.stream_sid} CallSid={self.call_sid}")

                elif event == "media":
                    if not self.stream_sid and msg.get("streamSid"):
                        self.stream_sid = msg.get("streamSid")

                    media = msg.get("media", {})
                    payload_b64 = media.get("payload", "")
                    if not payload_b64:
                        continue

                    mulaw_chunk = base64.b64decode(payload_b64)
                    pcm_chunk = TwilioAudioTranscoder.mulaw_to_pcm16k(mulaw_chunk)
                    chunk_duration = len(mulaw_chunk) / 8000.0

                    # Neural VAD evaluation (filters out phone line hiss, breathing, ambient background noise)
                    vad_state, vad_conf = await self.vad.analyze_pcm_chunk(pcm_chunk)

                    # Neural Barge-In: ONLY interrupt when verified human speech confidence >= 0.75
                    if self.is_agent_speaking:
                        if vad_state == VADState.SPEAKING and vad_conf >= 0.75:
                            print(f"🛑 [Twilio Neural Barge-In] Confirmed user speech (conf={vad_conf:.2f})! Yielding turn...")
                            self.interrupted = True
                            self.is_agent_speaking = False
                            # Clear Twilio's audio playback buffer in caller earpiece immediately
                            if self.stream_sid:
                                await self.ws.send_json({
                                    "event": "clear",
                                    "streamSid": self.stream_sid
                                })
                            self.speech_detected = True
                            self.pcm_buffer.clear()
                            self.pcm_buffer.extend(pcm_chunk)
                            self.consecutive_silence_sec = 0.0
                        continue

                    if vad_state in [VADState.STARTING, VADState.SPEAKING]:
                        if not self.speech_detected:
                            self.speech_detected = True
                            print(f"📞 [Twilio] Caller speaking (conf={vad_conf:.2f})...")
                        self.pcm_buffer.extend(pcm_chunk)
                        self.consecutive_silence_sec = 0.0
                    elif self.speech_detected:
                        self.pcm_buffer.extend(pcm_chunk)
                        self.consecutive_silence_sec += chunk_duration

                        if self.consecutive_silence_sec >= self.silence_threshold or vad_state == VADState.STOPPING:
                            self.speech_detected = False
                            self.consecutive_silence_sec = 0.0
                            print(f"📞 [Twilio] Pause of {self.silence_threshold}s detected, processing reply...")
                            asyncio.create_task(self._process_turn(bytes(self.pcm_buffer)))
                            self.pcm_buffer.clear()

                elif event == "stop":
                    print(f"📞 [Twilio] Call ended (StreamSid={self.stream_sid})")
                    self.is_active = False
                    break

        except Exception as e:
            if "disconnect" in str(e).lower():
                print(f"📞 [Twilio] WebSocket disconnected (StreamSid={self.stream_sid})")
            else:
                print(f"⚠️ [Twilio] Stream error: {e}")
        finally:
            self.is_active = False

    async def _process_turn(self, pcm_bytes: bytes):
        """Runs STT -> LLM -> TTS pipeline and streams reply back to the caller."""
        if not pcm_bytes or len(pcm_bytes) < 3200:
            return

        self.interrupted = False
        self.is_agent_speaking = True
        try:
            audio_np = TwilioAudioTranscoder.pcm_to_float32(pcm_bytes)
            user_text = await asyncio.to_thread(self.stt.transcribe, audio_np)

            if not user_text or not user_text.strip():
                print("📞 [Twilio] Inaudible audio segment; skipping.")
                self.is_agent_speaking = False
                return

            print(f"📞 [Twilio] Caller: \"{user_text}\"")

            reply_text = await asyncio.to_thread(self.llm.chat, user_text)
            print(f"🤖 [Twilio] AI Reply: \"{reply_text}\"")

            tts_chunks = await asyncio.to_thread(lambda: list(self.tts.voice.synthesize(reply_text)))
            tts_pcm = b"".join(c.audio_int16_bytes for c in tts_chunks)

            mulaw_audio = TwilioAudioTranscoder.tts_pcm_to_mulaw8k(tts_pcm, self.tts.sample_rate)

            chunk_size = 160
            for i in range(0, len(mulaw_audio), chunk_size):
                if not self.is_active or self.interrupted:
                    print("🛑 [Twilio Barge-In] Audio playback halted due to caller interruption!")
                    break
                chunk = mulaw_audio[i : i + chunk_size]
                payload = base64.b64encode(chunk).decode("utf-8")
                media_msg = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": payload},
                }
                await self.ws.send_json(media_msg)
                await asyncio.sleep(0.018)

            if self.is_active and self.stream_sid:
                await self.ws.send_json({
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": "reply_complete"}
                })

        except Exception as e:
            print(f"⚠️ [Twilio] Error processing turn: {e}")
        finally:
            self.is_agent_speaking = False
