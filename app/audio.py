import os
import sys
import wave
import time
import shutil
import collections
import threading
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import sounddevice as sd
import onnxruntime as ort

from app.config import (
    AUDIO_SAMPLE_RATE,
    ALSA_INPUT_DEVICE,
    ALSA_OUTPUT_DEVICE,
    AUDIO_DIR,
    VAD_MODEL_PATH,
    VAD_THRESHOLD,
    SILENCE_DURATION_THRESHOLD,
)


def flush_stdin():
    """Flush any buffered characters from standard input."""
    try:
        import termios, select
        while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)
    except Exception:
        pass


def unmute_speakers():
    """Ensure laptop speakers and master volume are unmuted and set to audible levels."""
    for ctl in ["Speaker", "Master", "Headphone"]:
        try:
            subprocess.run(
                ["amixer", "-c", "2", "set", ctl, "85%"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def detect_audio_devices() -> Tuple[Optional[int], Optional[int]]:
    """
    Intelligently detects working capture (microphone) and playback (speaker) devices.
    Returns (input_device_index, output_device_index).
    """
    input_idx = None
    output_idx = None
    try:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            name = dev.get("name", "").lower()
            max_in = dev.get("max_input_channels", 0)
            max_out = dev.get("max_output_channels", 0)

            if max_in > 0:
                if any(k in name for k in ["acp", "dmic", "mic", "analog", "default"]):
                    if input_idx is None or "acp" in name or "dmic" in name:
                        input_idx = idx
                elif input_idx is None:
                    input_idx = idx

            if max_out > 0:
                if any(k in name for k in ["speaker", "analog", "alc", "default"]):
                    if output_idx is None or "speaker" in name or "analog" in name:
                        output_idx = idx
                elif output_idx is None:
                    output_idx = idx

        if input_idx is None and hasattr(sd, "default") and sd.default.device:
            default_in = sd.default.device[0]
            if default_in is not None and default_in >= 0:
                input_idx = default_in

        if output_idx is None and hasattr(sd, "default") and sd.default.device:
            default_out = sd.default.device[1]
            if default_out is not None and default_out >= 0:
                output_idx = default_out
    except Exception as e:
        print(f"⚠️ [Audio] Could not query audio devices: {e}")

    return input_idx, output_idx


class SileroVAD:
    """
    State-of-the-art Voice Activity Detector using Silero VAD v5 ONNX model.
    Processes 512-sample (32ms at 16kHz) chunks with a rolling 64-sample context (576 input).
    """
    def __init__(self, model_path: Path = VAD_MODEL_PATH, threshold: float = VAD_THRESHOLD):
        self.model_path = Path(model_path)
        self.threshold = threshold
        if not self.model_path.exists():
            raise FileNotFoundError(f"Silero VAD model not found at {self.model_path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.sr = np.array(16000, dtype=np.int64)
        self.reset_states()

    def reset_states(self) -> None:
        """Reset internal RNN state and context between utterances."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Evaluate speech probability for a 512-sample float32 chunk.
        """
        if audio_chunk.ndim == 1:
            audio_chunk = np.expand_dims(audio_chunk, axis=0)

        # Pad or slice to exactly 512 samples
        if audio_chunk.shape[1] < 512:
            pad = np.zeros((1, 512 - audio_chunk.shape[1]), dtype=np.float32)
            audio_chunk = np.concatenate([audio_chunk, pad], axis=1)
        elif audio_chunk.shape[1] > 512:
            audio_chunk = audio_chunk[:, :512]

        # Silero VAD v5 requires 576 samples (64 rolling context + 512 new chunk)
        input_tensor = np.concatenate([self._context, audio_chunk], axis=1).astype(np.float32)

        out, self._state = self.session.run(
            None,
            {"input": input_tensor, "state": self._state, "sr": self.sr}
        )

        self._context = input_tensor[:, -64:]
        return float(out[0][0])

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        return self.get_speech_probability(audio_chunk) >= self.threshold


def process_raw_audio_to_16k_mono(raw_frames: np.ndarray, native_sr: int = 48000) -> np.ndarray:
    """
    Converts raw stereo or mono audio into calibrated, DC-offset free 16kHz mono int16.
    """
    if raw_frames.ndim == 2:
        mono = raw_frames.mean(axis=1)
    else:
        mono = raw_frames

    # Remove DC bias
    mono = mono.astype(np.float64)
    mono_ac = mono - np.mean(mono)

    # Resample to 16000 Hz (decimate by 3 if 48000Hz)
    if native_sr == 48000:
        audio_16k = mono_ac[::3]
    elif native_sr == 16000:
        audio_16k = mono_ac
    else:
        new_len = int(len(mono_ac) * 16000 / native_sr)
        audio_16k = np.interp(np.linspace(0, len(mono_ac), new_len), np.arange(len(mono_ac)), mono_ac)

    # Normalize with safety headroom
    max_val = np.max(np.abs(audio_16k))
    if max_val > 1.0:
        audio_16k = audio_16k / max_val
    elif max_val > 0.0001:
        audio_16k = audio_16k / max(max_val, 0.1) * 0.9

    audio_int16 = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16)
    return audio_int16


class AudioRecorder:
    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE, alsa_device: Optional[str] = None):
        self.sample_rate = sample_rate
        self.alsa_device = alsa_device
        self.input_device, self.output_device = detect_audio_devices()
        dev_info = {}
        if self.input_device is not None:
            try:
                dev_info = sd.query_devices(self.input_device)
            except Exception:
                dev_info = {}
        self.native_sr = 48000
        self.channels = min(dev_info.get("max_input_channels", 2), 2) if dev_info else 1
        dev_name = dev_info.get("name", "Default") if dev_info else "None/Headless"
        print(f"🎤 Audio Capture Device: [{self.input_device}] {dev_name} ({self.channels}ch @ {self.native_sr}Hz)")

    def record_fixed_duration(self, output_path: Optional[Path] = None, duration: float = 1.5) -> Path:
        """
        Records audio from the microphone for a fixed duration in seconds.
        """
        if output_path is None:
            output_path = AUDIO_DIR / "fixed_record.wav"
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.input_device is None:
            silence = np.zeros(int(self.sample_rate * duration), dtype=np.int16)
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(silence.tobytes())
            return output_path

        recorded_blocks = []
        def audio_callback(indata, frames, time_info, status):
            recorded_blocks.append(indata.copy())

        try:
            stream = sd.InputStream(
                samplerate=self.native_sr,
                channels=self.channels,
                dtype="float32",
                device=self.input_device,
                callback=audio_callback,
                blocksize=1536,
            )
            with stream:
                time.sleep(duration)
        except Exception as e:
            print(f"⚠️ [AudioRecorder] Fixed recording note: {e}")

        if not recorded_blocks:
            raw_audio = np.zeros((int(self.native_sr * duration), self.channels), dtype=np.float32)
        else:
            raw_audio = np.concatenate(recorded_blocks, axis=0)

        audio_int16 = process_raw_audio_to_16k_mono(raw_audio, self.native_sr)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return output_path

    def listen_and_record_speech(
        self,
        output_path: Optional[Path] = None,
        silence_seconds: float = 3.0,
        vad_threshold: float = 0.40,
        max_speech_duration: float = 60.0,
    ) -> Optional[Path]:
        """
        Hands-free conversational listening:
        - Listens indefinitely for the user to speak (no buttons required).
        - Once speech starts, records all frames.
        - Automatically detects when user goes silent for `silence_seconds` (default: 3.0s).
        - Cuts the recording and returns the Path for processing.
        """
        if output_path is None:
            output_path = AUDIO_DIR / "vad_input.wav"
        else:
            output_path = Path(output_path)

        vad = SileroVAD(threshold=vad_threshold)
        blocksize_48k = 1536  # 1536 / 3 = 512 samples at 16kHz (32ms per chunk)
        chunk_time_sec = 0.032

        # Pre-roll buffer: preserve ~400ms of audio before speech onset
        pre_roll_chunks = int(0.4 / chunk_time_sec)
        pre_roll = collections.deque(maxlen=pre_roll_chunks)

        recorded_16k_chunks = []
        is_speaking = False
        consecutive_speech_chunks = 0
        consecutive_silence_chunks = 0
        chunks_for_silence = int(silence_seconds / chunk_time_sec)

        audio_queue = []
        lock = threading.Lock()
        stop_stream = threading.Event()

        def stream_callback(indata, frames, time_info, status):
            if not stop_stream.is_set():
                with lock:
                    audio_queue.append(indata.copy())

        stream = sd.InputStream(
            samplerate=self.native_sr,
            channels=self.channels,
            dtype="float32",
            device=self.input_device,
            callback=stream_callback,
            blocksize=blocksize_48k,
        )

        print("\n👂 [LISTENING] Speak naturally into your microphone...", flush=True)
        print(f"   (I will automatically answer after {silence_seconds:.1f} seconds of silence)\n", flush=True)

        speech_start_time = 0.0
        last_display_time = 0.0

        with stream:
            while True:
                chunk_48k = None
                with lock:
                    if audio_queue:
                        chunk_48k = audio_queue.pop(0)

                if chunk_48k is None:
                    time.sleep(0.01)
                    continue

                # Downmix 48kHz to 16kHz mono float32
                mono_48k = chunk_48k.mean(axis=1)
                mono_ac = mono_48k - np.mean(mono_48k)
                chunk_16k = mono_ac[::3].astype(np.float32)

                # Evaluate speech probability and energy
                prob = vad.get_speech_probability(chunk_16k)
                energy = float(np.std(chunk_16k))

                # Hybrid speech detection: high VAD confidence or moderate VAD with energy
                chunk_has_speech = (prob >= vad_threshold) or (prob > 0.25 and energy > 0.08)

                if not is_speaking:
                    pre_roll.append(chunk_16k)
                    if chunk_has_speech:
                        consecutive_speech_chunks += 1
                        if consecutive_speech_chunks >= 2:
                            is_speaking = True
                            speech_start_time = time.time()
                            print("🎙️  [USER SPEAKING] Voice detected! Listening to your query...", flush=True)
                            recorded_16k_chunks.extend(pre_roll)
                    else:
                        consecutive_speech_chunks = 0
                else:
                    recorded_16k_chunks.append(chunk_16k)

                    # Enforce max speech duration limit
                    if time.time() - speech_start_time > max_speech_duration:
                        print(f"\n⏹️  [MAX DURATION REACHED] {max_speech_duration}s speech limit reached.\n", flush=True)
                        stop_stream.set()
                        break

                    if not chunk_has_speech and prob < (vad_threshold * 0.7):
                        consecutive_silence_chunks += 1
                        silence_elapsed = consecutive_silence_chunks * chunk_time_sec

                        # Live countdown indicator every 0.3s
                        if time.time() - last_display_time > 0.3:
                            last_display_time = time.time()
                            percent = min(100, int((silence_elapsed / silence_seconds) * 100))
                            bar_len = 15
                            filled = int(bar_len * percent / 100)
                            bar = "█" * filled + "░" * (bar_len - filled)
                            print(f"\r⏳ [PAUSE DETECTED] Silence: {silence_elapsed:.1f}s / {silence_seconds:.1f}s [{bar}]", end="", flush=True)

                        if consecutive_silence_chunks >= chunks_for_silence:
                            print(f"\n⏹️  [FINISHED] {silence_seconds:.1f}s silence reached! Thinking & speaking...\n", flush=True)
                            stop_stream.set()
                            break
                    else:
                        if consecutive_silence_chunks > 10:
                            print("\n🎙️  [RESUMED] Voice detected again, continuing...", flush=True)
                        consecutive_silence_chunks = 0

        if not recorded_16k_chunks:
            return None

        # Concatenate and normalize
        full_audio_16k = np.concatenate(recorded_16k_chunks, axis=0)
        max_val = np.max(np.abs(full_audio_16k))
        if max_val > 0.0001:
            full_audio_16k = full_audio_16k / max(max_val, 0.1) * 0.9

        audio_int16 = (np.clip(full_audio_16k, -1.0, 1.0) * 32767).astype(np.int16)

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return output_path

    def record_push_to_talk(self, output_path: Optional[Path] = None) -> Path:
        """Push-to-talk mode."""
        if output_path is None:
            output_path = AUDIO_DIR / "mic_input.wav"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        flush_stdin()
        input("\n👉 Press [Enter] to START speaking...")
        flush_stdin()

        stop_event = threading.Event()
        recorded_blocks = []

        def audio_callback(indata, frames, time_info, status):
            if not stop_event.is_set():
                recorded_blocks.append(indata.copy())

        stream = sd.InputStream(
            samplerate=self.native_sr,
            channels=self.channels,
            dtype="float32",
            device=self.input_device,
            callback=audio_callback,
            blocksize=1536,
        )

        start_time = time.time()
        print("🎙️  [RECORDING] Speak now into your microphone! Press [Enter] when done.", end="", flush=True)

        with stream:
            time.sleep(0.3)
            flush_stdin()
            try:
                input()
            finally:
                stop_event.set()

        total_elapsed = time.time() - start_time
        print(f"\n⏹️  Finished recording ({total_elapsed:.1f}s). Processing audio...")

        if not recorded_blocks:
            raw_audio = np.zeros((16000, 1), dtype=np.float32)
        else:
            raw_audio = np.concatenate(recorded_blocks, axis=0)

        audio_int16 = process_raw_audio_to_16k_mono(raw_audio, self.native_sr)

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return output_path


def play_audio_file(wav_path: Path, alsa_device: str = ALSA_OUTPUT_DEVICE) -> None:
    """
    Play a WAV file through speakers using aplay.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"File not found: {wav_path}")

    if shutil.which("aplay"):
        try:
            subprocess.run(
                ["aplay", "-q", "-D", alsa_device, str(wav_path)],
                check=True,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            subprocess.run(["aplay", "-q", str(wav_path)], check=True)
