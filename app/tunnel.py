import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("voice_agent.tunnel")

BASE_DIR = Path(__file__).resolve().parent.parent
CLOUDFLARED_PATH = BASE_DIR / ".venv" / "bin" / "cloudflared"


class TunnelManager:
    _instance: Optional["TunnelManager"] = None

    def __init__(self, port: Optional[int] = None):
        if port is None:
            port = int(os.getenv("PORT", 8080))
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.public_url: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(cls, port: Optional[int] = None) -> "TunnelManager":
        if cls._instance is None:
            cls._instance = cls(port)
        elif port is not None and cls._instance.port != port:
            cls._instance.port = port
        return cls._instance

    def start(self) -> Optional[str]:
        """Starts public tunnel (Cloudflare quick tunnel with SSH fallback)."""
        if self.public_url and self.process and self.process.poll() is None:
            return self.public_url

        if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("CI") == "true" or os.getenv("ENVIRONMENT") == "test":
            self.public_url = "https://mock-tunnel.example.com"
            return self.public_url

        # Ensure previous dead process is stopped
        self.stop()

        # 1. Cloudflare Tunnel
        binary = str(CLOUDFLARED_PATH) if CLOUDFLARED_PATH.exists() else shutil.which("cloudflared")
        if binary:
            try:
                logger.info("Starting Cloudflare tunnel on port %s...", self.port)
                self.process = subprocess.Popen(
                    [binary, "tunnel", "--url", f"http://localhost:{self.port}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                start_time = time.time()
                while time.time() - start_time < 30:
                    if self.process.poll() is not None:
                        logger.warning("cloudflared exited with code %s", self.process.poll())
                        break

                    line = self.process.stdout.readline()
                    if not line:
                        time.sleep(0.1)
                        continue

                    cleaned_line = line.strip()
                    if "trycloudflare.com" in cleaned_line or "INF" in cleaned_line or "ERR" in cleaned_line:
                        logger.debug("cloudflared: %s", cleaned_line)

                    match = re.search(r'https://(?!api\.)[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if match:
                        self.public_url = match.group(0)
                        logger.info("Cloudflare tunnel active: %s", self.public_url)
                        self._thread = threading.Thread(target=self._drain_output, daemon=True)
                        self._thread.start()
                        return self.public_url
            except Exception as e:
                logger.warning("Cloudflare tunnel start failed: %s", e)
                self.stop()

        # 2. Fallback to SSH tunnel (localhost.run)
        try:
            logger.info("Attempting fallback SSH tunnel (localhost.run)...")
            self.process = subprocess.Popen(
                [
                    "ssh",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "ServerAliveInterval=30",
                    "-R", f"80:localhost:{self.port}",
                    "nokey@localhost.run",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            start_time = time.time()
            while time.time() - start_time < 20:
                if self.process.poll() is not None:
                    break
                line = self.process.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue

                if "https://" in line and ".lhr.life" in line:
                    for part in line.strip().split():
                        if part.startswith("https://") and ".lhr.life" in part:
                            self.public_url = part.strip(",")
                            logger.info("SSH tunnel active: %s", self.public_url)
                            self._thread = threading.Thread(target=self._drain_output, daemon=True)
                            self._thread.start()
                            return self.public_url
        except Exception as e:
            logger.warning("SSH tunnel start failed: %s", e)
            self.stop()

        return self.public_url

    def _drain_output(self):
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
            except Exception:
                break

    def stop(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
            self.public_url = None

    def restart(self) -> Optional[str]:
        self.stop()
        time.sleep(1)
        return self.start()

    def get_status(self) -> dict:
        is_alive = self.process is not None and self.process.poll() is None
        return {
            "running": is_alive,
            "public_url": self.public_url if is_alive else None,
            "type": "cloudflare" if (self.public_url and "trycloudflare.com" in self.public_url) else "ssh" if self.public_url else "none",
            "port": self.port,
        }
