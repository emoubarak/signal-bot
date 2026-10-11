"""Turn stored messages into what the model reads."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from llmbot.store import KIND_IMAGE, KIND_VOICE, StoredMessage

_DAYS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]

NameResolver = Callable[[StoredMessage], str | None]


def fmt_time(ts_ms: int, tz: ZoneInfo, now: datetime | None = None) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz)
    now = now or datetime.now(tz)
    if dt.date() == now.date():
        return dt.strftime("%H:%M")
    if (now - dt).days < 6:
        return f"{_DAYS[dt.weekday()]} {dt.strftime('%H:%M')}"
    return dt.strftime("%d/%m %H:%M")


def format_message(m: StoredMessage, tz: ZoneInfo, reply_name: NameResolver | None = None) -> str:
    prefix = {KIND_VOICE: "🎤 ", KIND_IMAGE: "🖼️ "}.get(m.kind, "")
    text = m.text.strip() or ("[image sans légende]" if m.kind == KIND_IMAGE else "")
    replied = reply_name(m) if (reply_name and m.reply_to_ts) else None
    who = m.author_name + (f" (répond à {replied})" if replied else "")
    return f"[{fmt_time(m.ts, tz)}] {who} : {prefix}{text}"


def build_payload(
    *,
    conversation: list[StoredMessage],
    target: StoredMessage | None,
    request: str | None,
    chat_name: str | None,
    tz: ZoneInfo,
    max_chars: int,
    reply_name: NameResolver | None = None,
) -> str:
    """[CONVERSATION] / [CIBLE] / [DEMANDE] blocks, oldest conversation lines
    dropped first to stay under `max_chars`."""
    fixed: list[str] = []
    if target is not None:
        fixed.append("[CIBLE]\n" + format_message(target, tz, reply_name))
    if request:
        fixed.append("[DEMANDE]\n" + request.strip())
    fixed_len = sum(len(p) + 2 for p in fixed)

    lines = [format_message(m, tz, reply_name) for m in conversation]
    header = "[CONVERSATION" + (f" — {chat_name}" if chat_name else "") + "]\n"
    budget = max_chars - fixed_len - len(header)
    while lines and sum(len(line) + 1 for line in lines) > budget:
        lines.pop(0)

    parts = []
    if lines:
        parts.append(header + "\n".join(lines))
    parts.extend(fixed)
    return "\n\n".join(parts)
