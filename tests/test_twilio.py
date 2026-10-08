import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import base64
import json
import numpy as np
from starlette.testclient import TestClient
from app.server import app
from app.twilio_handler import TwilioService, TwilioAudioTranscoder

client = TestClient(app)

def test_twilio_service():
    ts = TwilioService()
    # Test TwiML generation
    twiml = ts.generate_twiml("voice.example.com")
    assert "<Stream" in twiml
    assert "/twiml/stream" in twiml

    # Test call execution with dummy number
    result = ts.make_call("+15551234567", host="voice.example.com")
    assert result["status"] in ("mock_mode", "success", "error")
    assert result["to"] == "+15551234567"

def test_audio_transcoding():
    # 20ms of silence in mulaw = 160 bytes of 0xFF
    mulaw_chunk = b"\xff" * 160
    pcm16k = TwilioAudioTranscoder.mulaw_to_pcm16k(mulaw_chunk)
    assert len(pcm16k) > 0

    float32_audio = TwilioAudioTranscoder.pcm_to_float32(pcm16k)
    assert isinstance(float32_audio, np.ndarray)
    assert float32_audio.dtype == np.float32

    # Synthesize dummy 22050Hz PCM and convert to 8000Hz mulaw
    pcm22k = np.zeros(22050, dtype=np.int16).tobytes()
    mulaw_out = TwilioAudioTranscoder.tts_pcm_to_mulaw8k(pcm22k, 22050)
    assert len(mulaw_out) == 8000

def test_twilio_api_endpoints():
    r = client.get("/api/twilio/status")
    assert r.status_code == 200
    data = r.json()
    assert "configured" in data

    r = client.post("/api/twilio/call", json={"to_phone": "+15559876543"})
    assert r.status_code == 200
    assert r.json()["to"] == "+15559876543"

    r = client.post("/twiml/incoming")
    assert r.status_code == 200
    assert "application/xml" in r.headers["content-type"]
    assert "<Response>" in r.text

def test_model_switching_endpoints():
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) >= 2

    # Switch to Nemotron
    nemotron = [m for m in models if "Nemotron" in m][0]
    r = client.post("/api/models/select", json={"model": nemotron})
    assert r.status_code == 200
    assert r.json()["current"] == nemotron

    # Switch back to Qwen
    qwen = [m for m in models if "qwen" in m][0]
    r = client.post("/api/models/select", json={"model": qwen})
    assert r.status_code == 200
    assert r.json()["current"] == qwen

def test_twilio_websocket():
    with client.websocket_connect("/twiml/stream") as ws:
        # 1. Connected
        ws.send_text(json.dumps({"event": "connected"}))

        # 2. Start
        ws.send_text(json.dumps({
            "event": "start",
            "start": {
                "streamSid": "MZ12345",
                "callSid": "CA12345",
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}
            }
        }))

        # 3. Media (silence chunk)
        payload = base64.b64encode(b"\xff" * 160).decode("utf-8")
        ws.send_text(json.dumps({
            "event": "media",
            "media": {"payload": payload}
        }))

        # 4. Stop
        ws.send_text(json.dumps({"event": "stop"}))

if __name__ == "__main__":
    test_twilio_service()
    test_audio_transcoding()
    test_twilio_api_endpoints()
    test_model_switching_endpoints()
    test_twilio_websocket()
    print("All Twilio & Model Switching tests passed successfully!")
