"""Reminders: the model sets them with the `remind` tool, this service posts them when they are due.

A reminder lives in the bot's database, so it survives a restart: one that came due while the bot was off is
posted at the next start, with an apology if it is very late."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from signalbot import ReadyHandler, SendMessage

from llmbot.store import KIND_BOT, Store, StoredMessage

logger = logging.getLogger("llmbot.reminders")

MAX_PENDING = 20       # per chat
MAX_DAYS = 366
MAX_MESSAGE = 400
POLL_SECONDS = 15
LATE_AFTER_MS = 15 * 60_000
# Pauses (seconds) before each new attempt at a reminder that could not be sent: about an hour in all, so a
# signal-rest that restarts for a while does not lose it. After the last one it is marked failed.
RETRY_PAUSES = (15, 30, 60, 120, 240, 480, 900, 1800)


def parse_when(at, in_minutes, tz: ZoneInfo, now: datetime) -> tuple[int | None, str]:
    """(due timestamp in ms, "") or (None, what to tell the model)."""
    try:
        if in_minutes not in (None, ""):
            # in UTC: on the local wall clock, "in 60 min" across the autumn change lasted 120 min
            due = (now.astimezone(timezone.utc) + timedelta(minutes=float(in_minutes))).astimezone(tz)
        elif at:
            due = datetime.fromisoformat(str(at).strip().replace(" ", "T"))
            due = due.replace(tzinfo=tz) if due.tzinfo is None else due.astimezone(tz)
        else:
            return None, "Give the time as `at` (YYYY-MM-DD HH:MM, the group's time) or as `in_minutes`."
    except (ValueError, OverflowError):
        return None, "Could not read that time. Use `at` as YYYY-MM-DD HH:MM, or `in_minutes` as a number."
    # compared in UTC: two datetimes of the same zone compare by wall clock, and the two 02:30 of
    # the autumn night would be taken for the same time
    due, now = due.astimezone(timezone.utc), now.astimezone(timezone.utc)
    if due <= now:
        return None, "That time is already past. Work it out again from the current date and time."
    if due > now + timedelta(days=MAX_DAYS):
        return None, "Too far away: reminders go up to one year ahead."
    return int(due.timestamp() * 1000), ""


def when_text(due_ts: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(due_ts / 1000, tz).strftime("%A %d/%m/%Y %H:%M")


class ReminderService(ReadyHandler):
    """Posts due reminders. Started once the bot is connected."""

    def __init__(self, store: Store, settings=None) -> None:
        self.store = store
        self.settings = settings  # bot name and number (the posted reminder is stored as the bot's), owner
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
                posted = await bot.messages.send(SendMessage(text=text), r["chat_id"])
            except Exception as e:  # signal-cli down, or the bot left the group: try again later, then give up
                await self._failed(bot, r, now_ms, e)
                continue
            self.store.finish_reminder(r["id"])
            self._store_posted(r, posted, text)
            sent += 1
        return sent

    def _store_posted(self, r, posted, text: str) -> None:
        """The posted reminder is one of the bot's messages: a reply to it reaches the bot."""
        timestamp = getattr(posted, "timestamp", None)
        if self.settings is None or not timestamp:
            return
        self.store.add(StoredMessage(chat_id=r["chat_id"], ts=int(timestamp), author_id=self.settings.phone_number,
                                     author_name=self.settings.bot_name, kind=KIND_BOT, text=text))

    async def _failed(self, bot, r, now_ms: int, error: Exception) -> None:
        attempts = r["attempts"] + 1
        if attempts <= len(RETRY_PAUSES):
            self.store.reminder_failed(r["id"], now_ms + RETRY_PAUSES[attempts - 1] * 1000)
            logger.warning("reminder #%s not sent (%s), attempt %d, next in %d s", r["id"], type(error).__name__,
                           attempts, RETRY_PAUSES[attempts - 1])
            return
        self.store.reminder_failed(r["id"], None)
        logger.error("reminder #%s given up after %d attempts (%s)", r["id"], attempts, type(error).__name__)
        owner = getattr(self.settings, "owner_number", "")
        if not owner:
            return
        try:
            await bot.messages.send(SendMessage(
                text=f"A reminder for {r['author_name']} could not be posted and was dropped: {r['message'][:200]}"), owner)
        except Exception as e:
            logger.warning("owner not told about reminder #%s (%s)", r["id"], type(e).__name__)
