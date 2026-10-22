import logging
import os
import re
import time
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
from app.security import StreamSentenceBuffer, sanitize_prompt_input

logger = logging.getLogger("voice_agent.twilio")


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
        self._phone_number = phone_number.strip() if phone_number else ""
        self.public_url = public_url.strip().rstrip("/") if public_url else ""

    @property
    def phone_number(self) -> str:
        return os.getenv("TWILIO_PHONE_NUMBER", "").strip() or self._phone_number

    @phone_number.setter
    def phone_number(self, value: str):
        self._phone_number = value.strip() if value else ""

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

        # Check active environment variable override
        env_pub = os.getenv("PUBLIC_WEBHOOK_URL", "").strip().rstrip("/")
        if env_pub and "localhost" not in env_pub and "127.0.0.1" not in env_pub:
            return env_pub

        tunnel = TunnelManager.get_instance()
        if tunnel.public_url and "localhost" not in tunnel.public_url and "127.0.0.1" not in tunnel.public_url:
            return tunnel.public_url

        if self.public_url and "localhost" not in self.public_url and "127.0.0.1" not in self.public_url:
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
        recording_status_url = f"{public_base.rstrip('/')}/twiml/recording-status{qs}"

        try:
            from twilio.rest import Client
            client = Client(self.account_sid, self.auth_token)
            from_phone = self.phone_number
            # Standard outbound call with full telemetry, call recording, and status callback webhooks
            try:
                call = client.calls.create(
                    to=to_phone,
                    from_=from_phone,
                    url=webhook_url,
                    record=True,
                    recording_status_callback=recording_status_url,
                    recording_status_callback_method="POST",
                    status_callback=status_callback_url,
                    status_callback_method="POST",
                    status_callback_event=["initiated", "ringing", "answered", "completed"],
                    time_limit=600,
                )
            except Exception as param_err:
                err_text = str(param_err).lower()
                # Twilio trial accounts disallow time_limit, status_callback, status_callback_method, status_callback_event.
                # When detected, retry immediately with allowed parameters (to, from_, url, record) which trial accounts allow.
                if any(k in err_text for k in ["disallowed", "trial", "parameter", "permission"]):
                    logger.info("Retrying call with simplified trial account parameters (to, from_, url, record)...")
                    try:
                        call = client.calls.create(
                            to=to_phone,
                            from_=from_phone,
                            url=webhook_url,
                            record=True,
                        )
                    except Exception as rec_err:
                        # Secondary fallback in case carrier prohibits record parameter on trial account
                        logger.warning("Trial account rejected record parameter (%s), retrying without record...", rec_err)
                        call = client.calls.create(
                            to=to_phone,
                            from_=from_phone,
                            url=webhook_url,
                        )
                else:
                    raise param_err

            return {
                "status": "success",
                "call_sid": call.sid,
                "to": to_phone,
                "agent_id": agent_id,
                "campaign_id": campaign_id,
                "webhook_url": webhook_url,
                "recorded": True,
                "message": f"Calling {to_phone}...",
            }
        except Exception as e:
            cleaned_err = clean_ansi(str(e))
            return {
                "status": "error",
                "message": f"Twilio call failed: {cleaned_err}",
                "to": to_phone,
            }

    def ensure_call_recording(self, call_sid: str) -> bool:
        """
        Ensures two-party mixed call recording is initiated for the specified call.
        Safe to call multiple times (checks for existing active recordings first).
        """
        if not self.is_configured() or not call_sid:
            return False
        try:
            from twilio.rest import Client
            client = Client(self.account_sid, self.auth_token)
            existing = client.calls(call_sid).recordings.list(limit=1)
            if existing:
                return True
            client.calls(call_sid).recordings.create(
                recording_channels="mono",
                recording_track="both",
            )
            logger.info("Explicitly initiated dual-party call recording for %s", call_sid)
            return True
        except Exception as e:
            logger.debug("Could not initiate call recording for %s: %s", call_sid, e)
            return False

    def get_call_recordings(self, call_sid: str) -> List[Dict[str, Any]]:
        """Queries Twilio REST API for recordings associated with a specific CallSid."""
        if not self.is_configured() or not call_sid:
            return []
        try:
            from twilio.rest import Client
            client = Client(self.account_sid, self.auth_token)
            recordings = client.calls(call_sid).recordings.list(limit=5)
            res = []
            for r in recordings:
                res.append({
                    "sid": r.sid,
                    "call_sid": r.call_sid,
                    "duration": getattr(r, "duration", 0),
                    "status": getattr(r, "status", "completed"),
                    "uri": getattr(r, "uri", ""),
                })
            return res
        except Exception as e:
            logger.debug("Could not fetch recordings for call %s: %s", call_sid, e)
            return []

    def download_recording(self, recording_sid: str, call_sid: str) -> Optional[str]:
        """
        Downloads a Twilio call recording audio file (.mp3) using account HTTP Basic Auth credentials,
        saves it to AUDIO_DIR / f'call_recording_{call_sid}.mp3', and returns the relative audio URL.
        """
        if not self.is_configured() or not recording_sid:
            return None

        from app.config import AUDIO_DIR
        target_filename = f"call_recording_{call_sid}.mp3"
        target_path = AUDIO_DIR / target_filename

        # If already downloaded, return relative URL immediately
        if target_path.exists() and target_path.stat().st_size > 500:
            return f"/api/audio/{target_filename}"

        media_url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Recordings/{recording_sid}.mp3"
        try:
            import requests
            resp = requests.get(media_url, auth=(self.account_sid, self.auth_token), timeout=25)
            if resp.status_code == 200 and len(resp.content) > 100:
                target_path.write_bytes(resp.content)
                logger.info(
                    "Successfully downloaded Twilio recording %s for call %s (%s bytes) to %s",
                    recording_sid,
                    call_sid,
                    len(resp.content),
                    target_filename,
                )
                return f"/api/audio/{target_filename}"
            else:
                logger.warning("Twilio recording download returned status %s (len: %s)", resp.status_code, len(resp.content))
        except Exception as e:
            logger.warning("Error downloading Twilio call recording %s: %s", recording_sid, e)

        return None


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
        self.session_start_time = time.time()
        self.max_duration_sec = 600.0  # Enforce 10-minute maximum call duration

        self.pcm_buffer = bytearray()
        self.speech_detected = False
        self.consecutive_silence_sec = 0.0
        self.current_turn_task: Optional[asyncio.Task] = None
        self.conversation_history: List[Dict[str, str]] = []
        # Pipecat Neural VAD to eliminate line static & background noise false interruptions
        self.vad = PipecatNeuralVAD(sample_rate=16000, confidence=0.75, start_secs=0.20, stop_secs=0.80, min_volume=0.06)

    async def handle(self):
        """Main WebSocket loop for Twilio streaming."""
        try:
            while self.is_active:
                # Enforce call duration ceiling
                if time.time() - self.session_start_time > self.max_duration_sec:
                    logger.info("Call reached maximum duration limit (%ds). Terminating session.", self.max_duration_sec)
                    break

                raw_msg = await self.ws.receive_text()
                msg = json.loads(raw_msg)
                event = msg.get("event")

                if event == "connected":
                    logger.info("Twilio WebSocket event: connected")

                elif event == "start":
                    start_data = msg.get("start", {})
                    self.stream_sid = start_data.get("streamSid") or msg.get("streamSid")
                    self.call_sid = start_data.get("callSid")
                    logger.info("Twilio Call started: StreamSid=%s CallSid=%s", self.stream_sid, self.call_sid)

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
                            logger.info("Twilio Neural Barge-In: Confirmed user speech (conf=%.2f)! Yielding turn...", vad_conf)
                            self.interrupted = True
                            self.is_agent_speaking = False
                            if self.current_turn_task and not self.current_turn_task.done():
                                self.current_turn_task.cancel()
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
                            logger.debug("Twilio: Caller speaking (conf=%.2f)...", vad_conf)
                        self.pcm_buffer.extend(pcm_chunk)
                        self.consecutive_silence_sec = 0.0
                    elif self.speech_detected:
                        self.pcm_buffer.extend(pcm_chunk)
                        self.consecutive_silence_sec += chunk_duration

                        if self.consecutive_silence_sec >= self.silence_threshold or vad_state == VADState.STOPPING:
                            self.speech_detected = False
                            self.consecutive_silence_sec = 0.0
                            logger.debug("Twilio: Silence threshold reached, processing turn reply...")
                            if self.current_turn_task and not self.current_turn_task.done():
                                self.current_turn_task.cancel()
                            self.current_turn_task = asyncio.create_task(self._process_turn(bytes(self.pcm_buffer)))
                            self.pcm_buffer.clear()

                elif event == "stop":
                    logger.info("Twilio Call ended (StreamSid=%s)", self.stream_sid)
                    self.is_active = False
                    break

        except Exception as e:
            if "disconnect" in str(e).lower():
                logger.info("Twilio WebSocket disconnected (StreamSid=%s)", self.stream_sid)
            else:
                logger.warning("Twilio stream error: %s", e)
        finally:
            self.is_active = False
            if self.current_turn_task and not self.current_turn_task.done():
                self.current_turn_task.cancel()

            # Record turns and trigger post-call analysis upon stream completion
            if self.conversation_history and self.call_sid:
                try:
                    from app.conversations import ConversationManager
                    from app.call_analysis import CallAnalyzer
                    conv_mgr = ConversationManager.get_instance()
                    conv_id = f"conv-call-{self.call_sid}"
                    for turn in self.conversation_history:
                        conv_mgr.record_turn(
                            role=turn.get("role", "user"),
                            text=turn.get("content", ""),
                            conversation_id=conv_id,
                            source="phone",
                            interaction_type="phone",
                            call_sid=self.call_sid,
                        )
                    analyzer = CallAnalyzer.get_instance()
                    analysis = analyzer.analyze_transcript(self.conversation_history, agent_name="Twilio Agent", llm_client=self.llm)
                    conv_mgr.update_analysis("apex-ai-innovations", conv_id, analysis)
                    logger.info("Twilio media stream post-call analysis saved for %s (Lead Score: %s)", conv_id, analysis.get("lead_score"))
                except Exception as ex:
                    logger.debug("Could not record Twilio media stream post-call analysis: %s", ex)

    async def _stream_pcm_to_websocket(self, pcm_bytes: bytes):
        """Encodes linear PCM to mu-law and streams 20ms frames to the Twilio WebSocket."""
        if not pcm_bytes or not self.stream_sid:
            return

        mulaw_audio = TwilioAudioTranscoder.tts_pcm_to_mulaw8k(pcm_bytes, self.tts.sample_rate)
        chunk_size = 160
        for i in range(0, len(mulaw_audio), chunk_size):
            if not self.is_active or self.interrupted:
                logger.info("Twilio audio playback halted due to caller interruption.")
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

    async def _process_turn(self, pcm_bytes: bytes):
        """Runs STT -> LLM -> streaming sentence TTS pipeline and streams reply back to the caller."""
        if not pcm_bytes or len(pcm_bytes) < 3200:
            return

        self.interrupted = False
        self.is_agent_speaking = True
        try:
            audio_np = TwilioAudioTranscoder.pcm_to_float32(pcm_bytes)
            user_text = await asyncio.to_thread(self.stt.transcribe, audio_np)

            if not user_text or not user_text.strip():
                logger.debug("Twilio: Inaudible audio segment; skipping.")
                self.is_agent_speaking = False
                return

            logger.info("Twilio Caller: '%s'", user_text)

            sanitized_input = sanitize_prompt_input(user_text)
            self.conversation_history.append({"role": "user", "content": sanitized_input})
            reply_text = await asyncio.to_thread(
                self.llm.chat, sanitized_input, history=self.conversation_history
            )
            self.conversation_history.append({"role": "assistant", "content": reply_text})
            logger.info("Twilio AI Reply: '%s'", reply_text)

            # Low-latency sentence-level TTS synthesis and streaming
            buffer = StreamSentenceBuffer()
            sentences = buffer.add_token(reply_text)
            sentences.extend(buffer.flush())
            if not sentences:
                sentences = [reply_text]

            for sentence in sentences:
                if not self.is_active or self.interrupted:
                    logger.info("Twilio sentence synthesis aborted due to caller interruption.")
                    break
                tts_chunks = await asyncio.to_thread(lambda s=sentence: list(self.tts.voice.synthesize(s)))
                tts_pcm = b"".join(c.audio_int16_bytes for c in tts_chunks)
                await self._stream_pcm_to_websocket(tts_pcm)

            if self.is_active and self.stream_sid and not self.interrupted:
                await self.ws.send_json({
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": "reply_complete"}
                })

        except Exception as e:
            logger.warning("Twilio turn processing error: %s", e)
        finally:
            self.is_agent_speaking = False
