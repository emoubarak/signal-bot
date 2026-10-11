"""Attachments on disk and audio conversion."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("llmbot.media")

_EXT = {
    "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif",
    "audio/aac": "aac", "audio/mp4": "m4a", "audio/mpeg": "mp3", "audio/ogg": "ogg",
    "audio/opus": "opus", "audio/wav": "wav", "audio/x-wav": "wav", "audio/flac": "flac",
}
AUDIO_FORMATS_DIRECT = {"wav", "mp3", "aac", "ogg", "flac", "m4a"}


def ext_for(content_type: str | None) -> str:
    return _EXT.get((content_type or "").lower(), "bin")


def is_image(content_type: str | None) -> bool:
    return (content_type or "").startswith("image/")


def is_audio(content_type: str | None) -> bool:
    return (content_type or "").startswith("audio/")


def attachments_dir(data_dir: Path) -> Path:
    return data_dir / "attachments"


def save_attachment(data_dir: Path, chat_id: str, ts: int, index: int, content_type: str | None, b64: str) -> Path:
    chat_hash = hashlib.sha1(chat_id.encode()).hexdigest()[:12]
    folder = attachments_dir(data_dir) / chat_hash
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{ts}_{index}.{ext_for(content_type)}"
    path.write_bytes(base64.b64decode(b64))
    return path


def cleanup_attachments(data_dir: Path, retention_days: int) -> int:
    """Delete attachment files older than `retention_days`. Returns the count."""
    root = attachments_dir(data_dir)
    if not root.exists():
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for path in root.rglob("*"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


async def to_mp3(audio: bytes) -> bytes:
    """Convert any audio (Signal voice notes are AAC) to a small mono MP3."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
        "-ac", "1", "-ar", "16000", "-b:a", "48k", "-f", "mp3", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(audio)
    if proc.returncode != 0 or not out:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace')[:200]}")
    return out


async def audio_for_model(audio: bytes, content_type: str | None) -> tuple[bytes, str]:
    """Return (bytes, format) ready for OpenRouter's `input_audio`."""
    if ffmpeg_available():
        try:
            return await to_mp3(audio), "mp3"
        except Exception as e:  # fall back to the raw file
            logger.warning("ffmpeg conversion failed, sending raw audio: %s", e)
    fmt = ext_for(content_type)
    return audio, (fmt if fmt in AUDIO_FORMATS_DIRECT else "aac")


MODEL_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 3_000_000


def image_for_model(data: bytes, content_type: str | None) -> tuple[bytes, str]:
    """Images as the models accept them: JPEG/PNG/WebP, at most ~3 MB and 2048 px.
    Anything else (GIF, HEIC, huge photos) is re-encoded with ffmpeg; on failure
    the original is returned."""
    ct = (content_type or "").lower()
    if ct in MODEL_IMAGE_TYPES and len(data) <= MAX_IMAGE_BYTES:
        return data, ct
    if not ffmpeg_available():
        return data, ct or "image/jpeg"
    try:
        out = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-frames:v", "1",
             "-vf", "scale='min(2048,iw)':'min(2048,ih)':force_original_aspect_ratio=decrease",
             "-q:v", "4", "-f", "image2", "-c:v", "mjpeg", "pipe:1"],
            input=data, capture_output=True, timeout=30, check=True,
        ).stdout
        if out:
            return out, "image/jpeg"
    except Exception as e:
        logger.warning("image re-encoding failed: %s", e)
    return data, ct or "image/jpeg"
