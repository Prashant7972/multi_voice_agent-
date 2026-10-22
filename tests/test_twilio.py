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
from app.auth import create_access_token

client = TestClient(app)

def test_twilio_service():
    ts = TwilioService()
    try:
        twiml = ts.generate_twiml("voice.example.com", mode="stream")
    except TypeError:
        # In case generate_twiml does not accept a mode keyword
        twiml = ts.generate_twiml("voice.example.com")

    # Accept either stream-based or gather-based TwiML
    assert ("<Stream" in twiml) or ("<Gather" in twiml)
    assert "<Response>" in twiml

    # Test call execution with dummy number
    result = ts.make_call("+your nunber", host="voice.example.com")
    assert result["status"] in ("mock_mode", "success", "error")
    assert result["to"] == "+1your caller"

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

    admin_token = create_access_token({"sub": "admin-1", "email": "admin@apexai.io", "role": "admin"})
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/api/twilio/call", json={"to_phone": "+phone number"}, headers=headers)
    assert r.status_code == 200
    assert r.json()["to"] == "phone numebr"

    r = client.post("/twiml/incoming")
    assert r.status_code == 200
    assert "application/xml" in r.headers["content-type"]
    assert "<Response>" in r.text

def test_model_switching_endpoints():
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) >= 2

    # Switch to second model
    model_b = models[1]
    r = client.post("/api/models/select", json={"model": model_b})
    assert r.status_code == 200
    assert r.json()["current"] == model_b

    # Switch back to first model
    model_a = models[0]
    r = client.post("/api/models/select", json={"model": model_a})
    assert r.status_code == 200
    assert r.json()["current"] == model_a

def test_twilio_websocket():
    with client.websocket_connect("/twiml/stream") as ws:
        # 1. Connected
        ws.send_text(json.dumps({"event": "connected"}))

        # 2. Start
        ws.send_text(json.dumps({
            "event": "start",
            "start": {
                "streamSid": "id",
                "callSid": "calld5",
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

def test_twilio_trial_parameter_fallback():
    from unittest.mock import MagicMock, patch
    ts = TwilioService(account_sid="your id", auth_token="token", phone_number="+phone")
    mock_call = MagicMock()
    mock_call.sid = "your id "

    calls_created = []

    def fake_create(**kwargs):
        calls_created.append(kwargs)
        if len(calls_created) == 1:
            raise Exception("Unable to create record: Invalid or disallowed parameters provided - trial accounts have limited parameter access")
        return mock_call

    with patch("twilio.rest.Client") as MockClient:
        instance = MockClient.return_value
        instance.calls.create.side_effect = fake_create

        res = ts.make_call("ypur umebr ", host="voice.example.com")
        assert res["status"] == "success"
        assert res["call_sid"] == "id"
        assert len(calls_created) == 2
        # First call attempted standard status callbacks
        assert "status_callback" in calls_created[0]
        # Fallback call strictly stripped status callbacks and time limits
        assert "status_callback" not in calls_created[1]
        assert "time_limit" not in calls_created[1]
        assert calls_created[1]["to"] == "your number "
        assert calls_created[1]["from_"] == "callers numnber "

def test_telephony_audio_config_and_twiml_interaction():
    from app.server import resolve_telephony_audio_config

    # 1. Config resolution for different locales
    cfg_in = resolve_telephony_audio_config("+yor number ")
    assert cfg_in["voice"] == "Polly.Kajal-Neural"
    assert cfg_in["language"] == "en-IN"

    cfg_us = resolve_telephony_audio_config("+callers umber")
    assert cfg_us["voice"] == "Polly.Joanna-Neural"
    assert cfg_us["language"] == "en-US"

    cfg_uk = resolve_telephony_audio_config("+442071234567")
    assert cfg_uk["voice"] == "Polly.Amy-Neural"
    assert cfg_uk["language"] == "en-GB"

    # 2. Incoming call to Indian number generates Indian neural voice & en-IN speech recognition
    r_inc = client.post("/twiml/incoming", data={"To": "+your mumber ", "From": "which from you want the phonen"})
    assert r_inc.status_code == 200
    assert "Polly.Kajal-Neural" in r_inc.text
    assert 'language="en-IN"' in r_inc.text
    assert 'bargeIn="false"' in r_inc.text

    # 3. Respond with speech returns conversational answer and continues interactive Gather
    r_resp = client.post("/twiml/respond", data={"SpeechResult": "Hello, how are you?", "To": "+your number "})
    assert r_resp.status_code == 200
    assert "<Response>" in r_resp.text
    assert "<Gather" in r_resp.text
    assert 'bargeIn="true"' in r_resp.text

    # 4. Respond with empty speech returns friendly conversational reprompt
    r_empty = client.post("/twiml/respond", data={"SpeechResult": "", "To": "your number "})
    assert r_empty.status_code == 200
    assert ("couldn't quite hear you" in r_empty.text) or ("say that again" in r_empty.text)


def test_incoming_call_logs_initial_greeting():
    from app.server import conversation_manager
    call_sid = "CA_greeting_test_8877"
    try:
        r = client.post("/twiml/incoming", data={"To": "+917972412266", "From": "+from yoyu wnat the nube r", "CallSid": call_sid})
        assert r.status_code == 200
        assert "<Response>" in r.text

        conv = None
        for c in conversation_manager.conversations:
            if c.get("call_sid") == call_sid or c.get("id") == f"conv-call-{call_sid}":
                conv = c
                break
        assert conv is not None
        assert len(conv.get("messages", [])) >= 1
        first_msg = conv["messages"][0]
        assert first_msg["role"] == "assistant"
        assert first_msg["source"] == "phone"
        assert "Hello" in first_msg["text"]
    finally:
        conversation_manager.conversations = [c for c in conversation_manager.conversations if c.get("call_sid") != call_sid and c.get("id") != f"conv-call-{call_sid}"]
        conversation_manager.save()


def test_download_recording_and_status_callback():
    from unittest.mock import MagicMock, patch
    from app.server import conversation_manager
    from app.twilio_handler import TwilioService
    from app.config import AUDIO_DIR

    test_sid = "CA_rec_test_9911"
    rec_sid = "RE_mock_recording_55"
    test_conv_id = f"conv-call-{test_sid}"

    try:
        # Setup test conversation
        conversation_manager.record_turn(
            role="assistant",
            text="Hello, how can I help you?",
            conversation_id=test_conv_id,
            org_id="apex-ai-innovations",
            call_sid=test_sid,
            source="phone",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"ID3\x03\x00\x00\x00" + b"\x00" * 600

        ts = TwilioService(account_sid="yur roekn", auth_token="test_token_12345", phone_number="+tour number")

        with patch("requests.get", return_value=mock_resp):
            url = ts.download_recording(rec_sid, test_sid)
            assert url == f"/api/audio/call_recording_{test_sid}.mp3"
            rec_file = AUDIO_DIR / f"call_recording_{test_sid}.mp3"
            assert rec_file.exists()
            assert rec_file.stat().st_size > 500

        # Test /twiml/recording-status webhook
        with patch("app.twilio_handler.TwilioService.download_recording", return_value=f"/api/audio/call_recording_{test_sid}.mp3"):
            r = client.post("/twiml/recording-status", data={"CallSid": test_sid, "RecordingSid": rec_sid, "RecordingDuration": "25"})
            assert r.status_code == 200
            assert "<Response/>" in r.text

        # Verify conversation updated with audio URL and recording SID
        conv = conversation_manager.get_conversation("apex-ai-innovations", test_conv_id)
        assert conv is not None
        assert conv.get("audio_url") == f"/api/audio/call_recording_{test_sid}.mp3"
        assert conv.get("recording_sid") == rec_sid
    finally:
        conversation_manager.conversations = [c for c in conversation_manager.conversations if c.get("id") != test_conv_id and c.get("call_sid") != test_sid]
        conversation_manager.save()
        rec_file = AUDIO_DIR / f"call_recording_{test_sid}.mp3"
        if rec_file.exists():
            try:
                rec_file.unlink()
            except Exception:
                pass


def test_audio_endpoint_returns_404_for_missing_file():
    r = client.get("/api/audio/non_existent_audio_file_123456.mp3")
    assert r.status_code == 404
    assert r.json().get("detail") == "Audio file not found"


if __name__ == "__main__":
    test_twilio_service()
    test_audio_transcoding()
    test_twilio_api_endpoints()
    test_model_switching_endpoints()
    test_twilio_websocket()
    test_twilio_trial_parameter_fallback()
    test_telephony_audio_config_and_twiml_interaction()
    test_incoming_call_logs_initial_greeting()
    test_download_recording_and_status_callback()
    test_audio_endpoint_returns_404_for_missing_file()
    print("All Twilio & Model Switching tests passed successfully!")
