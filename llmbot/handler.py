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

from signalbot import DataMessageContext, DataMessageHandler, EditMessage, SendMessage

from llmbot import prompts, syncflag
from llmbot.agent import Agent, AgentRequest
from llmbot.personas import Registry
from llmbot.config import Settings
from llmbot.media import audio_for_model, is_audio, is_image, save_attachment
from llmbot.openrouter import OpenRouter, OpenRouterError, data_url
from llmbot.store import KIND_BOT, KIND_IMAGE, KIND_TEXT, KIND_VOICE, Store, StoredMessage
from llmbot.syncflag import is_sync

logger = logging.getLogger("llmbot.handler")

USAGE_CHAT = "chat"
USAGE_IMAGE = "image"
USAGE_TRANSCRIBE = "transcribe"
USAGE_DESCRIBE = "describe"


class LLMHandler(DataMessageHandler):
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
        names = [settings.bot_name, *settings.bot_aliases]
        self._name_re = re.compile(
            r"(?<![\w@])(" + "|".join(re.escape(_fold(n)) for n in names if n.strip()) + r")(?![\w@])", re.I)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def chat_model(self) -> str:
        return self.store.get_setting("chat_model") or self.settings.chat_model

    # ── entry point ─────────────────────────────────────────────────────────

    async def handle_data_message(self, c: DataMessageContext) -> None:
        msg = c.message
        if isinstance(msg, EditMessage):
            self._handle_edit(c)
            return

        async with self._locks[self._chat_id(c)]:
            rec = await self._record(c)
        if rec is not None and self._addressed_to_bot(c, rec):
            await self._converse(c, rec)

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
        name = msg.source_name or msg.source_number or self.store.author_name(self._chat_id(c), author_id) or "Inconnu"
        return author_id, name

    async def _record(self, c: DataMessageContext) -> StoredMessage | None:
        msg = c.message
        chat_id = self._chat_id(c)
        author_id, author_name = self._author(c)

        # Our own replies are stored when sent; a sync echo must not overwrite them.
        existing = self.store.get(chat_id, msg.timestamp)
        if existing is not None and existing.is_bot:
            return None

        text = (msg.text or "").strip()
        attachments: list[dict] = []
        transcript: str | None = None
        for i, att in enumerate(msg.attachments or []):
            if not att.base64_content or msg.view_once:  # view-once media is not kept
                continue
            path = save_attachment(self.settings.data_dir, chat_id, msg.timestamp, i, att.content_type, att.base64_content)
            attachments.append({"path": str(path), "content_type": att.content_type or ""})
            if att.is_voice_note and is_audio(att.content_type) and transcript is None:
                transcript = await self._transcribe(c, path, att.content_type)

        images = [a for a in attachments if is_image(a["content_type"])]
        if transcript is not None:
            kind = KIND_VOICE
            text = f"{transcript} — {text}" if text else transcript
        elif images:
            kind = KIND_IMAGE
            if self.settings.describe_images:
                description = await self._describe_image(c, Path(images[0]["path"]), images[0]["content_type"])
                if description:
                    text = f"{text} [image : {description}]" if text else f"[image : {description}]"
        else:
            kind = KIND_TEXT
            if not text and attachments:
                text = "[fichier " + ", ".join(a["content_type"] or "?" for a in attachments) + "]"

        if not text and not attachments:
            return None  # stickers, reactions, empty data messages

        rec = StoredMessage(
            chat_id=chat_id, ts=msg.timestamp, author_id=author_id, author_name=author_name,
            kind=kind, text=text, reply_to_ts=msg.quote.id if msg.quote else None, attachments=attachments,
        )
        self.store.add(rec)
        logger.info("[%s] %s (%s): %s", chat_id[:12], author_name, kind, text[:80])
        return rec

    def _handle_edit(self, c: DataMessageContext) -> None:
        msg = c.message
        assert isinstance(msg, EditMessage)
        author_id = msg.source_uuid or msg.source_number or "?"
        if msg.text and self.store.update_text(self._chat_id(c), msg.target_sent_timestamp, author_id, msg.text.strip()):
            logger.info("edited message %s", msg.target_sent_timestamp)

    async def _transcribe(self, c: DataMessageContext, path: Path, content_type: str | None) -> str:
        try:
            audio, fmt = await audio_for_model(path.read_bytes(), content_type)
            result = await self.llm.transcribe(
                model=self.settings.transcribe_model, audio=audio, audio_format=fmt, prompt=prompts.TRANSCRIBE,
            )
            self._usage(c, USAGE_TRANSCRIBE, result.model, result.cost)
            return result.text or "[inaudible]"
        except Exception as e:
            logger.exception("transcription failed")
            return f"[vocal non transcrit : {type(e).__name__}]"

    async def _describe_image(self, c: DataMessageContext, path: Path, content_type: str) -> str | None:
        try:
            result = await self.llm.chat(
                model=self.chat_model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompts.DESCRIBE_IMAGE},
                    {"type": "image_url", "image_url": {"url": data_url(path.read_bytes(), content_type)}},
                ]}],
                temperature=0.2, max_tokens=120,
            )
            self._usage(c, USAGE_DESCRIBE, result.model, result.cost)
            return result.text
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
        name = self.store.author_name(rec.chat_id, author_id) or quote.author_number or "Quelqu'un"
        if quote.author_number == self.settings.phone_number:
            name = self.settings.bot_name
        return StoredMessage(chat_id=rec.chat_id, ts=quote.id, author_id=author_id, author_name=name, text=quote.text or "")

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

    def _usage(self, c: DataMessageContext, kind: str, model: str, cost: float) -> None:
        msg = c.message
        self.store.add_usage(self._chat_id(c), msg.source_uuid or msg.source_number or "?", kind, model, cost)

    async def _within_limit(self, c: DataMessageContext, rec: StoredMessage, kind: str) -> bool:
        if self._is_self(c):
            return True
        limit, window, label = self.settings.rate_limit_per_hour, 3_600_000, "demandes par heure"
        if limit <= 0:
            return True
        used = self.store.count_usage(rec.author_id, kind, int(time.time() * 1000) - window)
        if used >= limit:
            await self._say(c, rec, f"Doucement : limite de {limit} {label} atteinte.")
            return False
        return True

    # ── conversation with the bot ───────────────────────────────────────────

    async def _converse(self, c: DataMessageContext, rec: StoredMessage) -> None:
        """Natural-language mode: the agent reads the request and decides what to do."""
        if not await self._within_limit(c, rec, USAGE_CHAT):
            return
        thread = self.store.conversation_thread(rec.chat_id, rec.ts) or [rec]
        quoted = self._quoted(c, rec)
        window = self.store.recent(rec.chat_id, self.settings.context_messages, before_ts=thread[0].ts)
        images_used = self.store.count_usage(rec.author_id, USAGE_IMAGE, int(time.time() * 1000) - 86_400_000)
        req = AgentRequest(
            chat_id=rec.chat_id, chat_name=self._chat_name(c), user_id=rec.author_id,
            is_admin=c.message.source_number in self.settings.admin_numbers,
            thread=thread, window=window, quoted=quoted,
            images_remaining=max(0, self.settings.images_per_day - images_used) if self.settings.images_per_day > 0 else 99,
        )
        await c.start_typing()
        try:
            result = await self.agent.run(req, chat_model=self.chat_model)
        except OpenRouterError as e:
            logger.error("agent failed: %s", e)
            await self._say(c, rec, f"Le modèle n'a pas répondu ({e}).")
            return
        except Exception:
            logger.exception("agent failed")
            await self._say(c, rec, "Erreur inattendue, réessaie dans un instant.")
            return
        finally:
            await c.stop_typing()
        self._usage(c, USAGE_CHAT, result.chat_model, result.chat_cost)
        for model, cost in result.image_costs:
            self._usage(c, USAGE_IMAGE, model, cost)
        if result.silent:
            return
        if result.images:
            await self._say(c, rec, result.text, attachments=result.images, kind=KIND_IMAGE,
                            stored_text=f"[image générée] {result.text}".strip())
        else:
            await self._say(c, rec, result.text)


def _fold(text: str) -> str:
    """Lower-case without accents, so « Léo » matches « leo »."""
    return "".join(ch for ch in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(ch) != "Mn")
