"""
Maintenance and Audio Storage Lifecycle Service
Provides periodic cleanup of temporary audio files and operational maintenance.
"""
import os
import time
import logging
from pathlib import Path
from typing import Optional
from app.config import AUDIO_DIR

logger = logging.getLogger("voice_agent.maintenance")


def cleanup_old_audio_files(max_age_hours: int = 24, base_dir: Path = AUDIO_DIR, audio_dir: Optional[Path] = None) -> int:
    """
    Purges temporary generated audio files older than `max_age_hours` from the audio directory.
    Target patterns: greeting_*, phone_reply_*, reply_*, input_*, current_*, old_*.
    Preserves active recordings and directory structure.
    Returns the count of purged files.
    """
    target_dir = Path(audio_dir) if audio_dir is not None else Path(base_dir)
    if not target_dir.exists():
        return 0

    now = time.time()
    cutoff_time = now - (max_age_hours * 3600)
    purged_count = 0

    temp_prefixes = ("greeting_", "phone_reply_", "reply_", "fixed_record", "current_input", "current_reply", "temp_", "old_")

    try:
        for file_path in target_dir.glob("*.wav"):
            if not file_path.is_file():
                continue

            # Only remove temporary auto-generated files matching temp prefixes
            if any(file_path.name.startswith(prefix) for prefix in temp_prefixes):
                try:
                    file_mtime = file_path.stat().st_mtime
                    if file_mtime < cutoff_time:
                        file_path.unlink(missing_ok=True)
                        purged_count += 1
                except Exception as e:
                    logger.debug("Could not purge temp file %s: %s", file_path.name, e)

    except Exception as e:
        logger.warning("Error during audio file cleanup: %s", e)

    if purged_count > 0:
        logger.info("Purged %d expired temporary audio files (>%dh old)", purged_count, max_age_hours)

    return purged_count
