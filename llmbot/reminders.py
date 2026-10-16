"""Reminders: the model sets them with the `remind` tool, this service posts them when they are due.

A reminder lives in the bot's database, so it survives a restart: one that came due while the bot was off is
posted at the next start, with an apology if it is very late."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from signalbot import ReadyHandler, SendMessage

from llmbot.store import Store

logger = logging.getLogger("llmbot.reminders")

MAX_PENDING = 20       # per chat
MAX_DAYS = 366
MAX_MESSAGE = 400
POLL_SECONDS = 15
LATE_AFTER_MS = 15 * 60_000


def parse_when(at, in_minutes, tz: ZoneInfo, now: datetime) -> tuple[int | None, str]:
    """(due timestamp in ms, "") or (None, what to tell the model)."""
    try:
        if in_minutes not in (None, ""):
            due = now + timedelta(minutes=float(in_minutes))
        elif at:
            due = datetime.fromisoformat(str(at).strip().replace(" ", "T"))
            due = due.replace(tzinfo=tz) if due.tzinfo is None else due.astimezone(tz)
        else:
            return None, "Give the time as `at` (YYYY-MM-DD HH:MM, the group's time) or as `in_minutes`."
    except (ValueError, OverflowError):
        return None, "Could not read that time. Use `at` as YYYY-MM-DD HH:MM, or `in_minutes` as a number."
    if due <= now:
        return None, "That time is already past. Work it out again from the current date and time."
    if due > now + timedelta(days=MAX_DAYS):
        return None, "Too far away: reminders go up to one year ahead."
    return int(due.timestamp() * 1000), ""


def when_text(due_ts: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(due_ts / 1000, tz).strftime("%A %d/%m/%Y %H:%M")


class ReminderService(ReadyHandler):
    """Posts due reminders. Started once the bot is connected."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self._task: asyncio.Task | None = None

    async def handle_ready(self, context) -> None:
        self._task = asyncio.create_task(self._loop(context.bot))

    async def _loop(self, bot) -> None:
        while True:
            try:
                await self.deliver_due(bot)
            except Exception:
                logger.exception("reminder loop failed")
            await asyncio.sleep(POLL_SECONDS)

    async def deliver_due(self, bot, now_ms: int | None = None) -> int:
        """Post every reminder that is due. Returns how many were sent."""
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        sent = 0
        for r in self.store.due_reminders(now_ms):
            text = "⏰ " + r["message"]
            if now_ms - r["due_ts"] > LATE_AFTER_MS:
                text += " (sorry, late: I was offline)"
            try:
                await bot.messages.send(SendMessage(text=text), r["chat_id"])
            except Exception as e:  # signal-cli down: try again next round, give up after a few
                self.store.reminder_failed(r["id"])
                logger.warning("reminder #%s not sent (%s)", r["id"], type(e).__name__)
                continue
            self.store.finish_reminder(r["id"])
            sent += 1
        return sent
