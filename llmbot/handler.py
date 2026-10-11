"""The single Signal handler: records every message, then runs commands or
answers when the bot is addressed. One handler so recording always happens
before the command reads the store."""
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
from llmbot.commands import (
    PromptCommand, Registry, parse_count_and_request, parse_image_args, split_command,
)
from llmbot.config import Settings
from llmbot.context import build_payload, format_message
from llmbot.media import audio_for_model, image_for_model, is_audio, is_image, save_attachment
from llmbot.openrouter import OpenRouter, OpenRouterError, data_url
from llmbot.store import (
    KIND_BOT, KIND_COMMAND, KIND_IMAGE, KIND_TEXT, KIND_VOICE, Store, StoredMessage,
)
from llmbot.syncflag import is_sync

logger = logging.getLogger("llmbot.handler")

USAGE_CHAT = "chat"
USAGE_IMAGE = "image"
USAGE_TRANSCRIBE = "transcribe"
USAGE_DESCRIBE = "describe"

SPECIAL_COMMANDS = {"aide", "help", "image", "edit", "transcris", "perso", "modele", "cout"}


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
        if rec is None:
            return

        parsed = split_command(rec.text) if rec.is_command else None
        if parsed:
            name, rest = parsed
            if name in SPECIAL_COMMANDS or self.registry.get(name):
                await self._dispatch(c, rec, name, rest)
            # other "/words" are not ours (another bot, a path…): stay silent
        elif self._addressed_to_bot(c, rec):
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
        if text.startswith("/"):
            kind = KIND_COMMAND  # an attached image stays available as /image reference
        elif transcript is not None:
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
        if self._is_self(c):
            return False
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
        """The message the command replies to, from the store or rebuilt from the quote."""
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
                   attachment: tuple[bytes, str] | None = None, attachments: list[tuple[bytes, str]] | None = None,
                   kind: str = KIND_BOT, stored_text: str | None = None) -> StoredMessage | None:
        files = list(attachments or []) + ([attachment] if attachment else [])
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
        if kind == USAGE_IMAGE:
            limit, window, label = self.settings.images_per_day, 86_400_000, "images par jour"
        else:
            limit, window, label = self.settings.rate_limit_per_hour, 3_600_000, "demandes par heure"
        if limit <= 0:
            return True
        used = self.store.count_usage(rec.author_id, kind, int(time.time() * 1000) - window)
        if used >= limit:
            await self._say(c, rec, f"Doucement : limite de {limit} {label} atteinte.")
            return False
        return True

    async def _ask(self, c: DataMessageContext, rec: StoredMessage, *, system: str,
                   messages: list[dict], temperature: float) -> str | None:
        """Run the chat model with typing indicator, rate limit and error handling.
        Returns the text, or None after having told the user what went wrong."""
        if not await self._within_limit(c, rec, USAGE_CHAT):
            return None
        full = [{"role": "system", "content": system.strip() + "\n" + prompts.COMMON_RULES.strip()}, *messages]
        await c.start_typing()
        try:
            result = await self.llm.chat(
                model=self.chat_model, messages=full, temperature=temperature,
                max_tokens=self.settings.max_output_tokens,
            )
        except OpenRouterError as e:
            logger.error("chat failed: %s", e)
            await self._say(c, rec, f"Le modèle n'a pas répondu ({e}).")
            return None
        except Exception:
            logger.exception("chat failed")
            await self._say(c, rec, "Erreur inattendue, réessaie dans un instant.")
            return None
        finally:
            await c.stop_typing()
        self._usage(c, USAGE_CHAT, result.model, result.cost)
        if not result.text:
            await self._say(c, rec, "Le modèle a renvoyé une réponse vide.")
            return None
        return result.text + (" […]" if result.truncated else "")

    # ── conversation with the bot ───────────────────────────────────────────

    async def _converse(self, c: DataMessageContext, rec: StoredMessage) -> None:
        """Natural-language mode: the agent reads the request and decides what to do."""
        if not await self._within_limit(c, rec, USAGE_CHAT):
            return
        thread = self.store.conversation_thread(rec.chat_id, rec.ts) or [rec]
        quoted = self._quoted(c, rec)
        window = self.store.recent(rec.chat_id, 30, before_ts=thread[0].ts)
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

    # ── commands ────────────────────────────────────────────────────────────

    async def _dispatch(self, c: DataMessageContext, rec: StoredMessage, name: str, rest: str) -> None:
        if name in ("aide", "help"):
            await self._cmd_help(c, rec)
        elif name in ("image", "edit"):
            await self._cmd_image(c, rec, rest, require_reference=(name == "edit"))
        elif name == "transcris":
            await self._cmd_transcribe(c, rec)
        elif name == "perso":
            await self._cmd_perso(c, rec, rest)
        elif name == "modele":
            await self._cmd_model(c, rec, rest)
        elif name == "cout":
            await self._cmd_cost(c, rec)
        else:
            cmd = self.registry.get(name)
            if cmd:
                await self._cmd_prompt(c, rec, cmd, rest)

    async def _cmd_help(self, c: DataMessageContext, rec: StoredMessage) -> None:
        lines = [f"Appelle-moi par mon nom ({self.settings.bot_name}), mentionne-moi, réponds à un de mes messages "
                 "ou écris-moi en privé, et dis ce que tu veux : un résumé, une explication, un roast, une image "
                 "(« fais une image de… », « mets un chapeau sur cette photo » en répondant à l'image), ce que dit "
                 "un vocal… Les raccourcis ci-dessous font pareil sans passer par moi.", "", "Raccourcis :"]
        for cmd in self.registry.builtin():
            lines.append(f"/{cmd.name} — {cmd.description}")
        lines += [
            "/perso <personnage> : <message> — réponse dans la peau d'un personnage",
            "/image [format] <description> — génère une image (formats : 1:1, 16:9, 9:16, 4:3…) ; en réponse à une image, la modifie",
            "/transcris — en réponse à un vocal, renvoie sa transcription",
            "/modele [slug|reset] — affiche ou change le modèle de chat",
            "/cout — dépenses du jour et du mois",
        ]
        custom = self.registry.custom()
        if custom:
            lines.append("")
            lines.append("Personnalisées :")
            lines += [f"/{cmd.name} — {cmd.description.removeprefix('prompt personnalisé : ')}" for cmd in custom]
        lines += [
            "",
            "Comment ça marche :",
            "- Répondre à un message avec une commande la fait porter sur ce message.",
            "- /commande 20 porte sur les 20 derniers messages.",
            "- /commande texte ajoute une consigne.",
            "- Les vocaux sont transcrits et compris comme des messages.",
            f"- Limites : {self.settings.rate_limit_per_hour} demandes par heure, {self.settings.images_per_day} images par jour, par personne.",
        ]
        await self._say(c, rec, "\n".join(lines))

    async def _cmd_prompt(self, c: DataMessageContext, rec: StoredMessage, cmd: PromptCommand, rest: str) -> None:
        n, request = parse_count_and_request(rest)
        max_ctx = self.settings.max_context_messages
        quoted = self._quoted(c, rec)
        target: StoredMessage | None = None

        if quoted is not None:
            target = quoted
            conversation = self.store.recent(rec.chat_id, 10, before_ts=quoted.ts)
        elif n is not None:
            conversation = self.store.recent(rec.chat_id, min(max(n, 1), max_ctx), before_ts=rec.ts)
        elif cmd.since_last:
            last = self.store.last_command_ts(rec.chat_id, f"/{cmd.name}", before_ts=rec.ts)
            conversation = self.store.recent(rec.chat_id, max_ctx, before_ts=rec.ts, after_ts=last)
            if last is not None:  # the answer to the previous call is not news
                conversation = [m for m in conversation if not (m.is_bot and m.reply_to_ts and m.reply_to_ts <= last)]
            if not conversation and last is not None:
                await self._say(c, rec, f"Rien de nouveau depuis le dernier /{cmd.name}.")
                return
        else:
            conversation = self.store.recent(rec.chat_id, cmd.default_window, before_ts=rec.ts)

        if cmd.needs_target and target is None and conversation:
            candidates = [m for m in conversation if not m.is_bot and m.author_id != rec.author_id] or \
                         [m for m in conversation if not m.is_bot] or conversation
            target = candidates[-1]
            conversation = [m for m in conversation if m.ts != target.ts]

        if not conversation and target is None and not request:
            await self._say(c, rec, f"Rien à traiter : réponds à un message, ou /{cmd.name} 20 pour les 20 derniers, "
                                    f"ou /{cmd.name} <texte>.")
            return

        payload = build_payload(
            conversation=conversation, target=target, request=request or None, chat_name=self._chat_name(c),
            tz=self.tz, max_chars=self.settings.max_input_chars, reply_name=self._reply_name,
        )
        answer = await self._ask(c, rec, system=cmd.prompt, messages=[{"role": "user", "content": payload}],
                                 temperature=cmd.temperature)
        if answer:
            await self._say(c, rec, answer)

    async def _cmd_perso(self, c: DataMessageContext, rec: StoredMessage, rest: str) -> None:
        if ":" not in rest:
            await self._say(c, rec, "Usage : /perso <personnage> : <message>\nEx. : /perso un pirate ivre : c'est quoi le sens de la vie ?")
            return
        persona, request = (p.strip() for p in rest.split(":", 1))
        quoted = self._quoted(c, rec)
        if not request and quoted is None:
            await self._say(c, rec, "Il me faut un message après « : », ou réponds à un message.")
            return
        conversation = self.store.recent(rec.chat_id, 10, before_ts=(quoted or rec).ts)
        payload = build_payload(conversation=conversation, target=quoted, request=request or None,
                                chat_name=self._chat_name(c), tz=self.tz, max_chars=self.settings.max_input_chars,
                                reply_name=self._reply_name)
        system = f"Personnage à incarner : {persona}\n\n{prompts.PERSO}"
        answer = await self._ask(c, rec, system=system, messages=[{"role": "user", "content": payload}], temperature=1.0)
        if answer:
            await self._say(c, rec, answer)

    async def _cmd_transcribe(self, c: DataMessageContext, rec: StoredMessage) -> None:
        quoted = self._quoted(c, rec)
        if quoted is None or quoted.kind != KIND_VOICE:
            await self._say(c, rec, "Réponds à un vocal avec /transcris.")
            return
        await self._say(c, rec, f"🎤 {quoted.author_name} : {quoted.text}")

    async def _cmd_image(self, c: DataMessageContext, rec: StoredMessage, rest: str, *, require_reference: bool) -> None:
        ratio, prompt = parse_image_args(rest)
        references: list[tuple[bytes, str]] = []
        quoted = self._quoted(c, rec)
        for source in (quoted, rec):
            if source is None:
                continue
            for img in source.images():
                path = Path(img["path"])
                if path.exists():
                    references.append(image_for_model(path.read_bytes(), img["content_type"]))
        if require_reference and not references:
            await self._say(c, rec, "Réponds à une image avec /edit <modification>.")
            return
        if not prompt:
            await self._say(c, rec, "Usage : /image [1:1|16:9|9:16|4:3…] <description>\n"
                                    "En réponse à une image : /image <modification à apporter>.")
            return
        if not await self._within_limit(c, rec, USAGE_IMAGE):
            return

        await c.start_typing()
        try:
            result = await self.llm.generate_image(
                model=self.settings.image_model, prompt=prompt,
                aspect_ratio=ratio or ("auto" if references else "1:1"), references=references[:4] or None,
            )
        except OpenRouterError as e:
            logger.error("image failed: %s", e)
            await self._say(c, rec, f"Image impossible ({e}).")
            return
        except Exception:
            logger.exception("image failed")
            await self._say(c, rec, "Erreur inattendue pendant la génération.")
            return
        finally:
            await c.stop_typing()
        self._usage(c, USAGE_IMAGE, result.model, result.cost)
        label = "image modifiée" if references else "image générée"
        await self._say(c, rec, "", attachment=(result.data, result.media_type), kind=KIND_IMAGE,
                        stored_text=f"[{label} : {prompt}]")

    async def _cmd_model(self, c: DataMessageContext, rec: StoredMessage, rest: str) -> None:
        if not rest:
            override = self.store.get_setting("chat_model")
            await self._say(c, rec, f"Chat : {self.chat_model}{' (par défaut : ' + self.settings.chat_model + ')' if override else ''}\n"
                                    f"Images : {self.settings.image_model}\nVocaux : {self.settings.transcribe_model}\n"
                                    "Changer : /modele <slug OpenRouter> ou /modele reset")
            return
        admins = self.settings.admin_numbers
        if admins and c.message.source_number not in admins and not self._is_self(c):
            await self._say(c, rec, "Seuls les admins peuvent changer le modèle.")
            return
        if rest.lower() == "reset":
            self.store.set_setting("chat_model", None)
            await self._say(c, rec, f"Modèle de chat remis à {self.settings.chat_model}.")
            return
        slug = rest.split()[0]
        try:
            models = await self.llm.models()
        except OpenRouterError as e:
            await self._say(c, rec, f"Impossible de vérifier le modèle ({e}).")
            return
        info = models.get(slug)
        if info is None:
            await self._say(c, rec, f"Modèle inconnu sur OpenRouter : {slug}")
            return
        if "text" not in info.output_modalities:
            await self._say(c, rec, f"{slug} ne produit pas de texte.")
            return
        self.store.set_setting("chat_model", slug)
        vision = "image" in info.input_modalities
        await self._say(c, rec, f"Modèle de chat : {slug} ({info.name}){'' if vision else ' — ne lit pas les images'}")

    async def _cmd_cost(self, c: DataMessageContext, rec: StoredMessage) -> None:
        now = datetime.now(self.tz)
        day = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        month = int((now - timedelta(days=30)).timestamp() * 1000)
        await self._say(c, rec,
                        f"Aujourd'hui : {self.store.cost_since(day, rec.chat_id):.3f} $ ici, {self.store.cost_since(day):.3f} $ au total\n"
                        f"30 jours : {self.store.cost_since(month, rec.chat_id):.2f} $ ici, {self.store.cost_since(month):.2f} $ au total")


def _fold(text: str) -> str:
    """Lower-case without accents, so « Léo » matches « leo »."""
    return "".join(ch for ch in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(ch) != "Mn")
