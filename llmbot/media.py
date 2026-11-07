"""Attachments on disk and audio conversion."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import shutil
import unicodedata
import subprocess
import tempfile
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


# A local speech model with full attention needs memory in proportion to the audio (Parakeet: ~0.45 GB a
# minute; a 5 min note was killed at its 2 GB cap, 2026-11-07): long notes go in pieces of 90 s at most,
# cut on a pause when there is one after the first 30 s of the piece.
CHUNK_SECONDS = 90.0
MIN_CHUNK_SECONDS = 30.0
_DURATION = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")
_SILENCE = re.compile(r"silence_start: (-?[\d.]+)|silence_end: ([\d.]+)")


def plan_cuts(duration: float, pauses: list[float], max_s: float = CHUNK_SECONDS,
              min_s: float = MIN_CHUNK_SECONDS) -> list[tuple[float, float]]:
    """(start, end) of each piece: at most `max_s` long, ending on the latest pause that leaves at least
    `min_s`, or at `max_s` when there is none."""
    pieces, start = [], 0.0
    while duration - start > max_s:
        fits = [p for p in pauses if start + min_s <= p <= start + max_s]
        end = max(fits) if fits else start + max_s
        pieces.append((start, end))
        start = end
    pieces.append((start, duration))
    return pieces


async def _ffmpeg(args: list[str]) -> tuple[str, int]:
    proc = await asyncio.create_subprocess_exec("ffmpeg", "-hide_banner", "-y", *args,
                                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    return err.decode(errors="replace"), proc.returncode or 0


async def mp3_pieces(audio: bytes) -> list[bytes]:
    """Any audio as small mono MP3, in pieces of CHUNK_SECONDS at most (see above), in order. On files, not
    pipes: an MP3 read from a pipe has no known duration, and an .m4a cannot be read from one at all."""
    with tempfile.TemporaryDirectory(prefix="voice-") as tmp:
        src, mp3 = Path(tmp) / "in", Path(tmp) / "all.mp3"
        src.write_bytes(audio)
        err, code = await _ffmpeg(["-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", "-b:a", "48k",
                                   "-f", "mp3", str(mp3)])
        if code or not mp3.exists() or not mp3.stat().st_size:
            raise RuntimeError(f"ffmpeg failed: {err[:200]}")
        log, _ = await _ffmpeg(["-i", str(mp3), "-af", "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"])
        m = _DURATION.search(log)
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
        if duration <= CHUNK_SECONDS:
            return [mp3.read_bytes()]
        pauses, start = [], None
        for s_, e_ in _SILENCE.findall(log):
            if s_:
                start = float(s_)
            elif start is not None:
                pauses.append((start + float(e_)) / 2)
                start = None
        pieces = []
        for i, (a, b) in enumerate(plan_cuts(duration, pauses)):
            out = Path(tmp) / f"{i}.mp3"
            err, code = await _ffmpeg(["-loglevel", "error", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", str(mp3),
                                       "-c", "copy", str(out)])
            if code or not out.exists() or not out.stat().st_size:
                raise RuntimeError(f"ffmpeg failed cutting {a:.0f}-{b:.0f} s: {err[:200]}")
            pieces.append(out.read_bytes())
    logger.info("voice note of %.0f s sent in %d pieces", duration, len(pieces))
    return pieces


def foreign_script(text: str) -> bool:
    """Whether most letters are outside the Latin alphabet: a speech model guessing its language alone
    heard "test, test, un, deux" as "Тест, тест, один, две" (2026-11-07)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    other = sum(1 for ch in letters if not unicodedata.name(ch, "").startswith("LATIN"))
    return other > len(letters) / 2


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


def image_for_signal(data: bytes, content_type: str | None) -> tuple[bytes, str]:
    """Generated images as every Signal client shows them: JPEG or PNG. WebP
    (Muse Image's output) is re-encoded to JPEG; on failure the original is returned."""
    ct = (content_type or "").lower()
    if ct in {"image/jpeg", "image/png"} or not ffmpeg_available():
        return data, ct or "image/png"
    try:
        out = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-frames:v", "1",
             "-q:v", "2", "-f", "image2", "-c:v", "mjpeg", "pipe:1"],
            input=data, capture_output=True, timeout=30, check=True,
        ).stdout
        if out:
            return out, "image/jpeg"
    except Exception as e:
        logger.warning("image re-encoding failed: %s", e)
    return data, ct
