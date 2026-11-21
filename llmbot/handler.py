"""The single Signal handler: records every message, then lets the agent
answer when the bot is addressed (by name, mention, reply or in private)."""
from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from signalbot import DataMessageContext, DataMessageHandler, EditMessage, RemoteDeleteContext, RemoteDeleteHandler, SendMessage

from llmbot import prompts, syncflag
from llmbot.agent import Agent, AgentRequest
from llmbot.personas import Registry
from llmbot.config import Settings
from llmbot.media import audio_for_model, foreign_script, is_audio, is_image, load_image, mp3_pieces, save_attachment
from llmbot.openrouter import OpenRouter, OpenRouterError, data_url
from llmbot.runtime_config import UUID_RE
from llmbot.store import KIND_BOT, KIND_IMAGE, KIND_TEXT, KIND_VOICE, Store, StoredMessage
from llmbot.syncflag import is_sync

logger = logging.getLogger("llmbot.handler")

USAGE_CHAT = "chat"
USAGE_IMAGE = "image"
USAGE_TRANSCRIBE = "transcribe"
USAGE_DESCRIBE = "describe"

# `/clear`, `/clear!`, `!clear` …: the command alone, never inside a word (`/clearance`).
_CLEAR_RE = re.compile(r"^[/!]clear\s*[!?.]*$")
# Images of one message (an album) described on arrival, at most.
MAX_DESCRIBED = 4
DESCRIBE_BUDGET = 60  # seconds for all the descriptions of one message
# A local model that has not started answering after this long is loading or busy: say so once.
LOADING_NOTICE_SECONDS = 20
LOADING_NOTICE = "One moment, the model is loading or busy."
# Window of the rate limit, in ms.
LIMIT_WINDOW_MS = 3_600_000
# The tag a description takes in a message's text (older rows have no description saved apart from it).
_IMAGE_TAG = re.compile(r"\[image: .*?\]")


def _compose(kind: str, caption: str, attachments: list[dict], legacy: str = "") -> str:
    """The text stored for a message: its caption plus what was read from its media (the description of
    each image, the transcription of a voice note), kept with the attachments so a new caption (an edit)
    does not erase them. `legacy`: the old text of a row saved before that, to take its image tags from."""
    if kind == KIND_VOICE:
        transcript = next((a["transcript"] for a in attachments if "transcript" in a), None)
        if transcript is not None:
            return f"{transcript} - {caption}" if caption else transcript
    elif kind == KIND_IMAGE:
        tags = [f"[image: {a['description']}]" for a in attachments if a.get("description")]
        tags = tags or _IMAGE_TAG.findall(legacy)
        return " ".join(part for part in (caption, *tags) if part)
    return caption


class LLMHandler(DataMessageHandler, RemoteDeleteHandler):
    def __init__(self, settings: Settings, store: Store, llm: OpenRouter, registry: Registry) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.registry = registry
        self.tz = ZoneInfo(settings.timezone)
        self.agent = Agent(settings, store, llm, registry, self.tz)
        # signalbot runs several consumers: serialise recording per chat so a
        # message is stored (voice note transcribed) before the next one is handled
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # answers run as background tasks: signalbot has only three consumers, and
        # an image request holding one for a minute would hold up the next messages
        self._tasks: set[asyncio.Task] = set()
        self._name_re = self._compile_names()
        # one request at a time per person, so the rate limit sees the previous one; and who was already told
        # to slow down in the current window
        self._person_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._warned: set[str] = set()
        self._answering: set[tuple[str, int]] = set()  # (chat, ts) of the requests being answered

    def _compile_names(self) -> re.Pattern:
        names = [self.settings.bot_name, *self.settings.bot_aliases]
        return re.compile(
            r"(?<!\w)(" + "|".join(re.escape(_fold(n)) for n in names if n.strip()) + r")(?![\w@])", re.I)

    def refresh(self) -> None:
        """Take in what the admin page changed that was derived once at startup: the names
        people call the bot by, and the timezone."""
        self._name_re = self._compile_names()
        self.tz = self.agent.tz = ZoneInfo(self.settings.timezone)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def chat_model(self) -> str:
        return self.settings.chat_model

    # ── entry point ─────────────────────────────────────────────────────────

    async def handle_data_message(self, c: DataMessageContext) -> None:
        msg = c.message
        if not self._may_talk(c):
            logger.info("private message from someone who is not allowed: ignored")
            return
        if isinstance(msg, EditMessage):
            await self._handle_edit(c)
            return

        late = self._late(c)
        async with self._locks[self._chat_id(c)]:
            rec = await self._record(c)
        if rec is None:
            return
        if rec.kind == KIND_IMAGE and self.settings.describe_images:
            # described after the message is recorded: the chat's lock is not held while a model reads the image
            self._spawn(self._describe_message(c, rec))
        if self._is_clear_command(c, rec):  # even late: the owner's /clear still applies, from its own time
            self._spawn(self._clear_context(c, rec))
            return
        if late:
            # Delivered after the bot was off: kept for summaries and search, never answered,
            # so a question from hours ago does not get a reply out of nowhere.
            logger.info("[%s] late by %d min: stored, not answered", rec.chat_id[:12], late // 60_000)
            return
        if self._addressed_to_bot(c, rec):
            self._spawn(self._converse(c, rec))

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    # ── who may talk to the bot ─────────────────────────────────────────────

    def _may_talk(self, c: DataMessageContext) -> bool:
        """Private messages only: the owner always; ALLOWED_NUMBERS (numbers or Signal ids, for people who hide
        their number) when it is set; otherwise, when ALLOWED_GROUPS is set, the members of those groups; else
        everyone. Groups are filtered by signalbot itself."""
        msg = c.message
        if not msg.is_private() or self._is_self(c):
            return True
        s = self.settings
        ids = {i for i in (msg.source_number, msg.source_uuid) if i}
        if s.owner_number and s.owner_number in ids:
            return True
        if s.allowed_numbers:
            return bool(ids & set(s.allowed_numbers))
        if s.allowed_groups:
            return self._in_allowed_group(c, ids)
        return True

    def _in_allowed_group(self, c: DataMessageContext, ids: set[str]) -> bool:
        wanted = set(self.settings.allowed_groups)
        chats: set[str] = set()
        for group in c.bot.groups:
            if {group.id, group.internal_id, group.name} & wanted:
                if ids & set(group.members or []):
                    return True
                chats.add(group.internal_id)
        # someone who hides their number is not named by the member list: seen writing there. Not for the others:
        # a person removed from the group must not keep access because of what they wrote before
        return not c.message.source_number and any(self.store.has_posted_in(i, chats) for i in ids)

    # ── a message deleted for everyone ──────────────────────────────────────

    async def handle_remote_delete(self, context: RemoteDeleteContext) -> None:
        """The author deleted a message for everyone: it leaves the history (the model, search and summaries
        never see it again) and its files are removed."""
        msg = context.message
        authors = {i for i in (msg.source_uuid, msg.source_number) if i}
        chat_id = msg.source_or_group_id()
        async with self._locks[chat_id]:  # after a message being recorded (a voice note is transcribed first)
            attachments = self.store.delete_message(chat_id, msg.timestamp, authors)
        if attachments is None:
            return
        for a in attachments:
            Path(a.get("path", "")).unlink(missing_ok=True)
        logger.info("message %s deleted by its author: removed", msg.timestamp)

    def _late(self, c: DataMessageContext) -> int:
        """How late (ms) a message reaches the bot after Signal's server got it, when that is
        past LATE_REPLY_MINUTES; 0 otherwise. Measured from the server's receipt time, not the
        sender's clock, so a phone set to the wrong time is never taken for a late message."""
        limit = self.settings.late_reply_minutes
        received = getattr(c.message, "server_received_timestamp", None)
        if limit <= 0 or not received:
            return 0
        delay = int(time.time() * 1000) - int(received)
        return delay if delay > limit * 60_000 else 0

    # ── /clear ──────────────────────────────────────────────────────────────

    def _is_clear_command(self, c: DataMessageContext, rec: StoredMessage) -> bool:
        """`/clear`, from the owner only.

        The owner is `OWNER_NUMBER`: nobody else can make the bot forget a chat.
        `/clear` counts alone or followed by a separator (`/clear!`), never inside a
        word (`/clearance`)."""
        if not self.settings.owner_number:
            return False
        if not self._is_owner(c):
            return False
        return _CLEAR_RE.match(_fold(rec.text).strip()) is not None

    def _is_owner(self, c: DataMessageContext) -> bool:
        msg = c.message
        owner = self.settings.owner_number
        candidates = {msg.source_number, msg.source_uuid}
        if msg.is_private():
            return owner in candidates
        # in a group the number identifies the owner; a self-sent (sync) message
        # comes from the bot's own account, which is the owner's phone
        return owner in candidates or (msg.source_number == self.settings.phone_number and self._is_self(c))

    async def _clear_context(self, c: DataMessageContext, rec: StoredMessage) -> None:
        hidden = self.store.clear_context(rec.chat_id, rec.ts)
        logger.info("[%s] context cleared by the owner (%d earlier messages hidden)", rec.chat_id[:12], hidden)
        await self._say(c, rec, "Done, starting fresh here: I forget everything said before. "
                                f"({hidden} messages hidden, nothing deleted.)")

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("answer failed", exc_info=task.exception())

    @property
    def busy(self) -> int:
        return len(self._tasks)

    async def drain(self) -> None:
        """Wait for the answers in progress (shutdown, tests)."""
        while self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # ── recording ───────────────────────────────────────────────────────────

    def _chat_id(self, c: DataMessageContext) -> str:
        return c.message.source_or_group_id()

    def _is_self(self, c: DataMessageContext) -> bool:
        """Sent from the bot's own account (its linked phone)."""
        return c.message.source_number == self.settings.phone_number or is_sync(c.message.timestamp)

    def _author(self, c: DataMessageContext) -> tuple[str, str]:
        msg = c.message
        author_id = msg.source_uuid or msg.source_number or "?"
        if self._is_self(c):
            return author_id, self.settings.bot_name
        name = msg.source_name or msg.source_number or self.store.author_name(self._chat_id(c), author_id) or "Unknown"
        return author_id, name

    async def _record(self, c: DataMessageContext) -> StoredMessage | None:
        msg = c.message
        chat_id = self._chat_id(c)
        author_id, author_name = self._author(c)

        # Our own replies are stored when sent; a sync echo must not overwrite them.
        existing = self.store.get(chat_id, msg.timestamp)
        if existing is not None and existing.is_bot:
            return None
        # Already recorded from this author: Signal delivered it again. Not transcribed,
        # described or answered a second time.
        if existing is not None and existing.author_id == author_id:
            return None

        text = self._text_of(chat_id, msg).strip()
        attachments: list[dict] = []
        transcript: str | None = None
        for i, att in enumerate(msg.attachments or []):
            if not att.base64_content or msg.view_once:  # view-once media is not kept
                continue
            path = save_attachment(self.settings.data_dir, chat_id, msg.timestamp, i, att.content_type, att.base64_content)
            stored_att = {"path": str(path), "content_type": att.content_type or ""}
            attachments.append(stored_att)
            if att.is_voice_note and is_audio(att.content_type) and transcript is None:
                transcript = await self._transcribe(c, path, att.content_type)
                stored_att["transcript"] = transcript  # kept apart: an edit of the caption must not lose it

        images = [a for a in attachments if is_image(a["content_type"])]
        if transcript is not None:
            kind = KIND_VOICE
            text = _compose(kind, text, attachments)
        elif images:
            kind = KIND_IMAGE  # described afterwards, in the background (_describe_message)
        else:
            kind = KIND_TEXT
            if not text and attachments:
                text = "[file " + ", ".join(a["content_type"] or "?" for a in attachments) + "]"

        if not text and not attachments:
            return None  # stickers, reactions, empty data messages

        rec = StoredMessage(
            chat_id=chat_id, ts=msg.timestamp, author_id=author_id, author_name=author_name,
            kind=kind, text=text, reply_to_ts=msg.quote.id if msg.quote else None, attachments=attachments,
        )
        self.store.add(rec)
        logger.info("[%s] %s (%s): %s", chat_id[:12], author_name, kind, text[:80])
        return rec

    def _text_of(self, chat_id: str, msg) -> str:
        """The text of a message with each mention (Signal puts U+FFFC there) written as "@Name"."""
        return self._with_mentions(chat_id, msg.text or "", msg.mentions)

    def _with_mentions(self, chat_id: str, text: str, mentions) -> str:
        if not text or not mentions:
            return text
        units = text.encode("utf-16-le")  # Signal counts a mention's range in UTF-16 units
        for m in sorted(mentions, key=lambda m: m.start, reverse=True):
            a, b = m.start * 2, (m.start + m.length) * 2
            if m.start < 0 or b > len(units):
                continue
            units = units[:a] + f"@{self._mention_name(chat_id, m)}".encode("utf-16-le") + units[b:]
        return units.decode("utf-16-le", errors="replace")

    def _mention_name(self, chat_id: str, m) -> str:
        """Who a mention points to: never a number."""
        if m.number == self.settings.phone_number or (syncflag.own_uuid and m.uuid == syncflag.own_uuid):
            return self.settings.bot_name
        # signal-cli fills a mention's name with the number, or the Signal id when the number is hidden: what the
        # bot saw this person write under comes first, the mention's own name only when it is a real one
        for key in (m.uuid, m.number):
            if key and (name := self.store.author_name(chat_id, key)):
                return name
        if m.name and not m.name.lstrip("+").isdigit() and not UUID_RE.match(m.name):
            return m.name
        return "someone"

    async def _handle_edit(self, c: DataMessageContext) -> None:
        """An edit replaces the caption only: the description of an image and the transcription of a voice note
        stay. An edit that newly calls the bot by name gets its answer, once."""
        msg = c.message
        assert isinstance(msg, EditMessage)
        if not msg.text:
            return
        chat_id = self._chat_id(c)
        author_id = msg.source_uuid or msg.source_number or "?"
        async with self._locks[chat_id]:
            old = self.store.get(chat_id, msg.target_sent_timestamp)
            if old is None or old.author_id != author_id or old.is_bot:
                return
            caption = self._text_of(chat_id, msg).strip()
            text = _compose(old.kind, caption, old.attachments, legacy=old.text) if old.kind != KIND_TEXT else caption
            if text == old.text or not self.store.update_message(chat_id, old.ts, author_id, text, old.attachments):
                return
            rec = StoredMessage(chat_id=chat_id, ts=old.ts, author_id=author_id, author_name=old.author_name,
                                kind=old.kind, text=text, reply_to_ts=old.reply_to_ts, attachments=old.attachments)
        logger.info("edited message %s", old.ts)
        if msg.is_private() or self._is_self(c) or self._late(c):
            return
        if not self._was_addressed(old) and self._addressed_to_bot(c, rec):
            self._spawn(self._converse(c, rec))

    def _was_addressed(self, old: StoredMessage) -> bool:
        """Whether the message, before its edit, already called the bot: by name, by a reply to one of its
        messages, or because it is being answered or was answered (a text or an image)."""
        if self._name_re.search(_fold(old.text)) or (old.chat_id, old.ts) in self._answering:
            return True
        if self.store.has_reply(old.chat_id, old.ts, self.settings.phone_number):
            return True
        target = self.store.get(old.chat_id, old.reply_to_ts) if old.reply_to_ts else None
        return target is not None and target.is_bot

    async def _transcribe(self, c: DataMessageContext, path: Path, content_type: str | None) -> str:
        if not self.settings.transcription_enabled:
            return "[voice note not transcribed: transcription disabled]"
        s = self.settings
        if s.transcribe_base_url:
            try:
                text = await self._transcribe_locally(path)
                if text:
                    self._usage(c, USAGE_TRANSCRIBE, "local speech server", 0.0)
                    return text
                logger.warning("local speech server heard nothing%s",
                               ", trying OpenRouter" if s.transcribe_model else "")
            except Exception as e:
                logger.warning("local transcription failed (%s)%s", e, ", trying OpenRouter" if s.transcribe_model else "")
                if not s.transcribe_model:
                    return f"[voice note not transcribed: {type(e).__name__}]"
            if not s.transcribe_model:
                return "[inaudible]"
        try:
            audio, fmt = await audio_for_model(path.read_bytes(), content_type)
            names = ", ".join([self.settings.bot_name, *self.settings.bot_aliases])
            prompt = (prompts.TRANSCRIBE + f"\nA bot in the group is called {names}: when that name is spoken, "
                      "write it exactly like this.")
            result = await self.llm.transcribe(
                model=self.settings.transcribe_model, audio=audio, audio_format=fmt, prompt=prompt,
            )
            self._usage(c, USAGE_TRANSCRIBE, result.model, result.cost)
            return result.text or "[inaudible]"
        except Exception as e:
            logger.exception("transcription failed")
            return f"[voice note not transcribed: {type(e).__name__}]"

    async def _transcribe_locally(self, path: Path) -> str:
        """The local speech server: MP3 (it reads WAV, FLAC or MP3, not Signal's AAC), in pieces of 90 s at
        most sent one after the other (its memory grows with the audio), each asked again on the retry
        server, told the language, when it came out in another alphabet."""
        s = self.settings
        texts = []
        for piece in await mp3_pieces(path.read_bytes()):
            text = await self.llm.transcribe_local(base_url=s.transcribe_base_url, audio=piece, audio_format="mp3")
            if text and foreign_script(text) and s.transcribe_retry_url:
                try:
                    again = await self.llm.transcribe_local(base_url=s.transcribe_retry_url, audio=piece,
                                                            audio_format="mp3", language=s.transcribe_language)
                except Exception as e:
                    logger.warning("retry speech server failed (%s), keeping: %s", e, text[:80])
                else:
                    if again and not foreign_script(again):
                        logger.info("voice note in another alphabet (%s), heard again: %s", text[:60], again[:60])
                        text = again
            texts.append(text)
        return " ".join(t for t in texts if t).strip()

    async def _describe_message(self, c: DataMessageContext, rec: StoredMessage) -> None:
        """Describe the images of a message already recorded, then add the sentences to its text. An album is
        several images: each one gets its sentence, within DESCRIBE_BUDGET in all."""
        if not await self._can_describe():
            return
        images, found = rec.images()[:MAX_DESCRIBED], {}

        async def describe_all() -> None:
            for img in images:
                if d := await self._describe_image(c, Path(img["path"]), img["content_type"], rec.text):
                    found[img["path"]] = d
        try:
            await asyncio.wait_for(describe_all(), DESCRIBE_BUDGET)
        except TimeoutError:
            logger.warning("image descriptions stopped after %d s", DESCRIBE_BUDGET)
        if not found:
            return
        async with self._locks[rec.chat_id]:  # quick: no model call under the lock
            current = self.store.get(rec.chat_id, rec.ts)
            if current is None or current.author_id != rec.author_id:  # deleted meanwhile
                return
            for a in current.attachments:
                if a.get("path") in found:
                    a["description"] = found[a["path"]]
            self.store.update_message(rec.chat_id, rec.ts, rec.author_id,
                                      _compose(KIND_IMAGE, current.text, current.attachments), current.attachments)

    async def _can_describe(self) -> bool:
        """Describing runs on the chat model: OpenRouter needs its key; a local server must
        already hold the model, so a photo nobody asked about never swaps another model out."""
        if not self.settings.local_chat:
            return bool(self.settings.openrouter_api_key)
        if await self.llm.model_loaded(self.chat_model):
            return True
        logger.info("image not described: %s is not loaded on the local server", self.chat_model)
        return False

    async def _describe_image(self, c: DataMessageContext, path: Path, content_type: str,
                              caption: str = "") -> str | None:
        try:
            loaded = await load_image(path.read_bytes(), content_type)  # HEIC, GIF, huge photos
            if loaded is None:
                logger.warning("image %s cannot be read by the model (%s): not described", path.name, content_type)
                return None
            data, media_type = loaded
            prompt = prompts.DESCRIBE_IMAGE + (f"\nIt was posted with this text: {caption[:500]}" if caption else "")
            result = await self.llm.chat(
                model=self.chat_model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url(data, media_type)}},
                ]}],
                temperature=0.2, max_tokens=160, retry_on_truncation=False,
            )
            self._usage(c, USAGE_DESCRIBE, result.model, result.cost)
            if result.truncated:  # half a sentence would mislead summaries and search
                logger.warning("image description cut by the token limit, dropped")
                return None
            return (result.text or "").strip() or None
        except Exception:
            logger.exception("image description failed")
            return None

    # ── addressing ──────────────────────────────────────────────────────────

    def _addressed_to_bot(self, c: DataMessageContext, rec: StoredMessage) -> bool:
        msg = c.message
        if self._is_self(c):  # from the bot's own phone: only an explicit call by name
            return msg.is_group() and bool(self._name_re.search(_fold(rec.text)))
        if msg.is_private():
            return True
        for m in msg.mentions or []:
            if m.number == self.settings.phone_number or (syncflag.own_uuid and m.uuid == syncflag.own_uuid):
                return True
        if self._name_re.search(_fold(rec.text)):
            return True
        if msg.quote is not None:
            if msg.quote.author_number == self.settings.phone_number:
                return True
            quoted = self.store.get(rec.chat_id, msg.quote.id)
            if quoted is not None and quoted.is_bot:
                return True
        return False

    def _quoted(self, c: DataMessageContext, rec: StoredMessage) -> StoredMessage | None:
        """The message the request replies to, from the store or rebuilt from the quote."""
        quote = c.message.quote
        if quote is None:
            return None
        stored = self.store.get(rec.chat_id, quote.id)
        if stored is not None:
            return stored
        author_id = quote.author_uuid or quote.author_number or quote.author or "?"
        name = self.store.author_name(rec.chat_id, author_id) or "Someone"  # never the number: it reaches the model
        if quote.author_number == self.settings.phone_number:
            name = self.settings.bot_name
        return StoredMessage(chat_id=rec.chat_id, ts=quote.id, author_id=author_id, author_name=name,
                             text=self._with_mentions(rec.chat_id, quote.text or "", quote.mentions))

    def _reply_name(self, m: StoredMessage) -> str | None:
        target = self.store.get(m.chat_id, m.reply_to_ts) if m.reply_to_ts else None
        return target.author_name if target else None

    def _chat_name(self, c: DataMessageContext) -> str | None:
        gi = c.message.group_info
        return gi.group_name if gi else None

    # ── sending ─────────────────────────────────────────────────────────────

    async def _say(self, c: DataMessageContext, rec: StoredMessage, text: str, *,
                   attachments: list[tuple[bytes, str]] | None = None,
                   kind: str = KIND_BOT, stored_text: str | None = None) -> StoredMessage | None:
        files = list(attachments or [])
        send = SendMessage(text=text)
        if files:
            send.base64_attachments = [data_url(*f) for f in files]
        try:
            sent = await c.reply(send)
        except Exception:
            logger.exception("send failed")
            return None
        stored: list[dict] = []
        for i, (data, media_type) in enumerate(files):
            path = save_attachment(self.settings.data_dir, rec.chat_id, sent.timestamp, i, media_type,
                                   data_url(data, media_type).split(",", 1)[1])
            stored.append({"path": str(path), "content_type": media_type})
        bot_msg = StoredMessage(
            chat_id=rec.chat_id, ts=sent.timestamp, author_id=self.settings.phone_number,
            author_name=self.settings.bot_name, kind=kind, text=stored_text if stored_text is not None else text,
            reply_to_ts=rec.ts, attachments=stored,
        )
        self.store.add(bot_msg)
        return bot_msg

    def _usage(self, c: DataMessageContext, kind: str, model: str, cost: float) -> int:
        msg = c.message
        return self.store.add_usage(self._chat_id(c), msg.source_uuid or msg.source_number or "?", kind, model, cost)

    async def _reserve(self, c: DataMessageContext, rec: StoredMessage, kind: str) -> int | None:
        """Count a request against the person's limit before the model is called, so a burst, and a request
        that fails, are counted too. Returns the usage row to fill in once the cost is known, or None when the
        limit is reached ("Slow down" is said once, until the person is back under the limit)."""
        limit = self.settings.rate_limit_per_hour
        if limit > 0 and not self._is_self(c):
            used = self.store.count_usage(rec.author_id, kind, int(time.time() * 1000) - LIMIT_WINDOW_MS)
            if used >= limit:
                if rec.author_id not in self._warned:
                    self._warned.add(rec.author_id)
                    await self._say(c, rec, f"Slow down: {limit} requests per hour limit reached.")
                return None
            self._warned.discard(rec.author_id)
        return self._usage(c, kind, self.chat_model, 0.0)

    # ── conversation with the bot ───────────────────────────────────────────

    async def _loading_notice(self, c: DataMessageContext) -> None:
        await asyncio.sleep(LOADING_NOTICE_SECONDS)
        try:
            await c.reply(SendMessage(text=LOADING_NOTICE))  # not stored: it is not an answer
        except Exception:
            logger.exception("loading notice failed")

    async def _converse(self, c: DataMessageContext, rec: StoredMessage) -> None:
        """Natural-language mode: the agent reads the request and decides what to do. One request at a time
        per person: the next one is counted against the limit only once this one is."""
        key = (rec.chat_id, rec.ts)
        self._answering.add(key)
        try:
            async with self._person_locks[rec.author_id]:
                await self._answer(c, rec)
        finally:
            self._answering.discard(key)

    async def _answer(self, c: DataMessageContext, rec: StoredMessage) -> None:
        slot = await self._reserve(c, rec, USAGE_CHAT)
        if slot is None:
            return
        thread = self.store.conversation_thread(rec.chat_id, rec.ts) or [rec]
        quoted = self._quoted(c, rec)
        window = self.store.recent(rec.chat_id, self.settings.context_messages, before_ts=thread[0].ts)
        images_used = self.store.count_usage(rec.author_id, USAGE_IMAGE, int(time.time() * 1000) - 86_400_000)
        req = AgentRequest(
            chat_id=rec.chat_id, chat_name=self._chat_name(c), user_id=rec.author_id,
            thread=thread, window=window, quoted=quoted,
            images_remaining=max(0, self.settings.images_per_day - images_used) if self.settings.images_per_day > 0 else 99,
            # after a /clear the model must not see, or reach back to, what was said before
            context_start=self.store.context_start(rec.chat_id),
        )
        await c.start_typing()
        # a local model can take minutes to load or to get its turn: tell the group once, not after the deadline
        notice = asyncio.create_task(self._loading_notice(c)) if self.settings.local_chat else None
        if notice is not None:
            req.on_model_reply = notice.cancel  # the model answered: it was not waiting any more
        try:
            result = await self.agent.run(req, chat_model=self.chat_model)
        except OpenRouterError as e:
            logger.error("agent failed: %s", e)
            await self._say(c, rec, f"The model did not answer ({e}).")
            return
        except Exception:
            logger.exception("agent failed")
            await self._say(c, rec, "Unexpected error, try again in a moment.")
            return
        finally:
            if notice is not None:
                notice.cancel()
            await c.stop_typing()
        self.store.update_usage(slot, result.chat_model, result.chat_cost)
        for model, cost in result.image_costs:
            self._usage(c, USAGE_IMAGE, model, cost)
        if result.silent:
            logger.info("[%s] stays silent: name only mentioned in passing", rec.chat_id[:12])
            return
        if result.images:
            label = "gif" if all(t == "image/gif" for _, t in result.images) else "generated image"
            await self._say(c, rec, result.text, attachments=result.images, kind=KIND_IMAGE,
                            stored_text=f"[{label}] {result.text}".strip())
        else:
            await self._say(c, rec, result.text)


def _fold(text: str) -> str:
    """Lower-case without accents, so "Leo" matches "leo"."""
    return "".join(ch for ch in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(ch) != "Mn")
