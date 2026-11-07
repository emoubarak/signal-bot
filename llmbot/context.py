"""Turn stored messages into what the model reads."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from llmbot.store import KIND_IMAGE, KIND_VOICE, StoredMessage

_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

NameResolver = Callable[[StoredMessage], str | None]


def fmt_time(ts_ms: int, tz: ZoneInfo, now: datetime | None = None) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz)
    now = now or datetime.now(tz)
    if dt.date() == now.date():
        return dt.strftime("%H:%M")
    if (now - dt).days < 6:
        return f"{_DAYS[dt.weekday()]} {dt.strftime('%H:%M')}"
    return dt.strftime("%d/%m %H:%M")


def format_message(m: StoredMessage, tz: ZoneInfo, reply_name: NameResolver | None = None,
                   with_time: bool = True) -> str:
    """One transcript line. Without the time for the lines the model reads as the live conversation:
    it once continued that "[06:38] Name: " format instead of answering (2026-11-06)."""
    prefix = {KIND_VOICE: "🎤 ", KIND_IMAGE: "🖼️ "}.get(m.kind, "")
    text = m.text.strip() or ("[image with no caption]" if m.kind == KIND_IMAGE else "")
    replied = reply_name(m) if (reply_name and m.reply_to_ts) else None
    who = m.author_name + (f" (replying to {replied})" if replied else "")
    line = f"{who}: {prefix}{text}"
    return f"[{fmt_time(m.ts, tz)}] {line}" if with_time else line
