"""Announce at startup, once the bot is connected: where it runs and on which model.

The model actually serving the request is asked, not the configured name: a
llama-swap selector such as `qwen3.8-flash-next-ablit` answers as whichever model it
loaded (`qwen3.8-flash-next-q2_0`), and that is what the owner wants to read.
"""
from __future__ import annotations

import logging
from pathlib import PurePath

from signalbot import ReadyHandler, SendMessage

from llmbot.config import Settings
from llmbot.openrouter import OpenRouter, OpenRouterError

logger = logging.getLogger("llmbot.startup")

# One token is enough to learn which model answered; the request itself is thrown away.
PROBE = "ping"


class StartupAnnouncer(ReadyHandler):
    """Says hello in private to the owner."""

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
        served = reply.model or self.settings.chat_model
        # llama-server names a model by its file path: announce "Qwen3.5-4B-Q4_K_M", not the whole path
        return PurePath(served).stem if served.endswith(".gguf") else served

    async def _announcement(self) -> str:
        model = await self._served_model()
        where = "locally" if self.settings.local_chat else "on OpenRouter"
        return f"I restarted, beep-boop, I am model {model} {where}."

    async def handle_ready(self, context) -> None:
        # The owner's private chat only: no group announcement, so restarts
        # stay silent in groups.
        if not self.settings.owner_number:
            logger.info("no startup announcement: no OWNER_NUMBER configured")
            return
        text = await self._announcement()
        try:
            await context.bot.messages.send(SendMessage(text=text), self.settings.owner_number)
        except Exception as e:
            logger.warning("startup announcement to %s failed: %s", self.settings.owner_number, e)
            return
        logger.info("startup announcement sent to %s: %s", self.settings.owner_number, text)
