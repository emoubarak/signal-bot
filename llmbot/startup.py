"""Announce at startup, once the bot is connected: where it runs and on which model.

The model actually serving the request is asked, not the configured name: a
llama-swap selector such as `qwen3.6-35b-ablit` answers as whichever model it
loaded (`qwen3.8-flash-next-q2_0`), and that is what the owner wants to read.
"""
from __future__ import annotations

import logging

from signalbot import ReadyHandler, SendMessage

from llmbot.config import Settings
from llmbot.openrouter import OpenRouter, OpenRouterError

logger = logging.getLogger("llmbot.startup")

# One token is enough to learn which model answered; the request itself is thrown away.
PROBE = "ping"


class StartupAnnouncer(ReadyHandler):
    """Says hello in every allowed group, and in private to the owner."""

    def __init__(self, settings: Settings, llm: OpenRouter) -> None:
        self.settings = settings
        self.llm = llm

    async def _served_model(self) -> str:
        """The model that really answers, or the configured name if it will not say."""
        try:
            reply = await self.llm.chat(
                model=self.settings.chat_model,
                messages=[{"role": "user", "content": PROBE}],
                temperature=0.0, max_tokens=1, retry_on_truncation=False,
            )
        except OpenRouterError as e:
            logger.warning("startup probe failed (%s), announcing the configured name", e)
            return self.settings.chat_model
        return reply.model or self.settings.chat_model

    async def _announcement(self) -> str:
        model = await self._served_model()
        where = "en local" if self.settings.local_chat else "sur OpenRouter"
        return f"J'ai redémarré, blipbloup, je suis le modèle {model} {where}."

    async def handle_ready(self, context) -> None:
        text = await self._announcement()
        # The owner's private chat, so a restart is visible even when no group is
        # configured: the bot writes to the number that owns it.
        recipients = list(self.settings.allowed_groups)
        if self.settings.owner_number:
            recipients.append(self.settings.owner_number)
        if not recipients:
            logger.info("no startup announcement: no group configured and no OWNER_NUMBER")
            return
        sent = 0
        for recipient in recipients:
            try:
                await context.bot.messages.send(SendMessage(text=text), recipient)
                sent += 1
            except Exception as e:  # one unreachable chat must not stop the others
                logger.warning("startup announcement to %s failed: %s", recipient, e)
        logger.info("startup announcement sent to %d/%d chat(s): %s", sent, len(recipients), text)
