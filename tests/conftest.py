"""Test harness: a SignalBot wired to the real handler, signal-cli-rest-api stubbed
(same patches as signalbot.test_utils.mock_chat), OpenRouter replaced by FakeLLM."""
from __future__ import annotations

import asyncio
import base64
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from signalbot import SignalBot
from signalbot.test_utils import ChatTestCase
from signalbot._generated import SendMessageResponse
from signalbot.test_utils.chat_testing import (
    AboutMock, CheckSignalServiceMock, Envelope, GetAllMock, ReactMock, ReceiveMock,
    RemoteDeleteMock, SendMock, _new_timestamp,
)

from llmbot.personas import Registry
from llmbot.config import Settings
from llmbot.handler import LLMHandler
from llmbot.openrouter import ChatResult, ImageResult, ModelInfo
from llmbot.store import Store

BOT_NUMBER = ChatTestCase.phone_number
GROUP_INTERNAL = ChatTestCase.group_internal_id
GROUP_ID = ChatTestCase.group_id
# What the handler uses as chat id for the test group (signalbot gives the internal id)
CHAT_ID = GROUP_INTERNAL

PAUL = ("+33611111111", "11111111-1111-1111-1111-111111111111", "Paul")
LEA = ("+33622222222", "22222222-2222-2222-2222-222222222222", "Lea")

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@dataclass
class FakeLLM:
    answer: str = "model reply"
    transcript: str = "hi this is a voice note"
    chat_calls: list[dict] = field(default_factory=list)
    transcribe_calls: list[dict] = field(default_factory=list)
    image_calls: list[dict] = field(default_factory=list)
    # Scripted replies for the agent: a str is a final answer, a dict {"name", "arguments"} a tool call
    script: list = field(default_factory=list)
    # When set, image generation waits for this event (to test concurrency)
    image_gate: asyncio.Event | None = None

    async def chat(self, *, model, messages, temperature=0.7, max_tokens=1000, tools=None, retry_on_truncation=True):
        self.chat_calls.append({"model": model, "messages": messages, "temperature": temperature, "tools": tools})
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, dict):
                call = {"id": f"call{len(self.chat_calls)}", "type": "function",
                        "function": {"name": step["name"], "arguments": json.dumps(step.get("arguments", {}))}}
                return ChatResult(text="", finish_reason="tool_calls", model=model, cost=0.0001,
                                  message={"role": "assistant", "content": None, "tool_calls": [call]}, tool_calls=[call])
            return ChatResult(text=step, finish_reason="stop", model=model, cost=0.0001,
                              message={"role": "assistant", "content": step})
        return ChatResult(text=self.answer, finish_reason="stop", model=model, cost=0.0001,
                          message={"role": "assistant", "content": self.answer})

    async def transcribe(self, *, model, audio, audio_format, prompt):
        self.transcribe_calls.append({"model": model, "format": audio_format, "size": len(audio)})
        return ChatResult(text=self.transcript, finish_reason="stop", model=model, cost=0.00005)

    async def generate_image(self, *, model, prompt, aspect_ratio="1:1", references=None):
        if self.image_gate is not None:
            await self.image_gate.wait()
        self.image_calls.append({"model": model, "prompt": prompt, "aspect_ratio": aspect_ratio,
                                 "references": references or []})
        return ImageResult(data=PNG_1PX, media_type="image/png", model=model, cost=0.018)

    async def models(self, refresh=False):
        return {
            "anthropic/claude-haiku-5.5": ModelInfo("anthropic/claude-haiku-5.5", "Haiku", ["text", "image"], ["text"]),
            "openai/gpt-5-mini": ModelInfo("openai/gpt-5-mini", "GPT-5 mini", ["text"], ["text"]),
        }

    def last_user_text(self) -> str:
        content = [m for m in self.chat_calls[-1]["messages"] if m["role"] == "user"][-1]["content"]
        if isinstance(content, str):
            return content
        return "\n".join(p["text"] for p in content if p.get("type") == "text")

    def last_system(self) -> str:
        return self.chat_calls[-1]["messages"][0]["content"]


def group_message(user, text, *, quote: dict | None = None, attachments: list[dict] | None = None,
                  mentions: list[dict] | None = None, ts: int | None = None, private: bool = False) -> Envelope:
    number, uid, name = user
    ts = ts or _new_timestamp()
    data = {"timestamp": ts, "message": text, "expiresInSeconds": 0, "viewOnce": False,
            "mentions": mentions or [], "attachments": attachments or []}
    if not private:
        data["groupInfo"] = {"groupId": GROUP_INTERNAL, "type": "DELIVER", "revision": 1}
    if quote:
        data["quote"] = quote
    env = {"account": BOT_NUMBER, "envelope": {
        "source": number, "sourceNumber": number, "sourceUuid": uid, "sourceName": name, "sourceDevice": 1,
        "timestamp": ts, "serverReceivedTimestamp": ts, "serverDeliveredTimestamp": ts, "dataMessage": data}}
    return Envelope(json.dumps(env))


def sync_message(text: str, *, to_user=None, ts: int | None = None) -> Envelope:
    """A message typed from the bot's own phone: in the test group, or in the
    private chat with `to_user`."""
    ts = ts or _new_timestamp()
    sent = {"timestamp": ts, "message": text, "expiresInSeconds": 0, "viewOnce": False,
            "destination": None, "destinationNumber": None, "destinationUuid": None}
    if to_user:
        sent.update({"destination": to_user[0], "destinationNumber": to_user[0], "destinationUuid": to_user[1]})
    else:
        sent["groupInfo"] = {"groupId": GROUP_INTERNAL, "type": "DELIVER", "revision": 1}
    env = {"account": BOT_NUMBER, "envelope": {
        "source": BOT_NUMBER, "sourceNumber": BOT_NUMBER, "sourceUuid": str(uuid.uuid4()), "sourceName": "Bot",
        "sourceDevice": 1, "timestamp": ts, "serverReceivedTimestamp": ts, "serverDeliveredTimestamp": ts,
        "syncMessage": {"sentMessage": sent}}}
    return Envelope(json.dumps(env))


def edit_message(user, new_text: str, target_ts: int) -> Envelope:
    number, uid, name = user
    ts = _new_timestamp()
    env = {"account": BOT_NUMBER, "envelope": {
        "source": number, "sourceNumber": number, "sourceUuid": uid, "sourceName": name, "sourceDevice": 1,
        "timestamp": ts, "serverReceivedTimestamp": ts, "serverDeliveredTimestamp": ts,
        "editMessage": {"targetSentTimestamp": target_ts, "dataMessage": {
            "timestamp": ts, "message": new_text, "expiresInSeconds": 0, "viewOnce": False,
            "groupInfo": {"groupId": GROUP_INTERNAL, "type": "DELIVER", "revision": 1}}}}}
    return Envelope(json.dumps(env))


def quote_of(user, ts: int, text: str) -> dict:
    number, uid, _ = user
    return {"id": ts, "author": uid, "authorNumber": number, "authorUuid": uid, "text": text}


def bot_quote(ts: int, text: str) -> dict:
    return {"id": ts, "author": BOT_NUMBER, "authorNumber": BOT_NUMBER, "authorUuid": str(uuid.uuid4()), "text": text}


def voice_attachment() -> dict:
    return {"id": "voice1.aac", "contentType": "audio/aac", "isVoiceNote": True, "size": 10}


def image_attachment() -> dict:
    return {"id": "pic1.png", "contentType": "image/png", "isVoiceNote": False, "size": len(PNG_1PX)}


class Harness:
    def __init__(self, mocker, tmp_path: Path, **settings_overrides) -> None:
        self.llm = FakeLLM()
        self.settings = Settings(
            signal_service=ChatTestCase.signal_service, phone_number=BOT_NUMBER, openrouter_api_key="x",
            bot_name="Jarvis", data_dir=tmp_path, **settings_overrides,
        )
        self.store = Store(":memory:")
        self.handler = LLMHandler(self.settings, self.store, self.llm, Registry.load(None))
        self.bot = SignalBot(ChatTestCase.config)
        self.bot.register(self.handler)

        mocker.patch("llmbot.media.ffmpeg_available", return_value=False)
        mocker.patch("signalbot._client.reactions.ReactionsClient.react", new_callable=ReactMock)
        self.send_mock = mocker.patch("signalbot._client.messages.MessagesClient.send", new_callable=SendMock)
        # bot replies dated now, not 2021, so ordering in the store is realistic
        self.send_mock.side_effect = lambda dm: [SendMessageResponse(timestamp=str(_new_timestamp()))
                                                 for _ in dm.recipients or [None]]
        self.receive_mock = mocker.patch("signalbot._client.messages.MessagesClient.receive", new_callable=ReceiveMock)
        mocker.patch("signalbot._client.groups.GroupsClient.get_all", new_callable=GetAllMock)
        mocker.patch("signalbot._client.general.GeneralClient.about", new_callable=AboutMock)
        mocker.patch("signalbot._client.SignalAPI.check_signal_service", new_callable=CheckSignalServiceMock)
        mocker.patch("signalbot._client.receipts.ReceiptsClient.send", new_callable=AsyncMock)
        mocker.patch("signalbot._client.messages.MessagesClient.start_typing", new_callable=AsyncMock)
        mocker.patch("signalbot._client.messages.MessagesClient.stop_typing", new_callable=AsyncMock)
        mocker.patch("signalbot._client.messages.MessagesClient.remote_delete", new_callable=RemoteDeleteMock)
        self.download_mock = mocker.patch(
            "signalbot._client.attachments.AttachmentsClient.download", new_callable=AsyncMock,
            return_value=base64.b64encode(PNG_1PX).decode(),
        )
        self._initialised = False

    async def run(self, *messages: Envelope, drain: bool = True) -> None:
        self.receive_mock.define(messages)
        if not self._initialised:
            await self.bot._async_init()
            self._initialised = True
        pipeline = self.bot._pipeline
        await pipeline._produce(1)
        while pipeline._q.qsize() > 0:
            await pipeline._consume_new_item(1)
            if drain:  # answer before the next message, as the tests expect
                await self.handler.drain()

    def sent(self) -> list:
        return self.send_mock.results()

    def sent_texts(self) -> list[str]:
        return [m.message for m in self.sent()]

    def bot_timestamps(self, chat_id: str) -> list[int]:
        rows = self.store._db.execute(
            "SELECT ts FROM messages WHERE chat_id=? AND author_id=? ORDER BY ts", (chat_id, BOT_NUMBER)).fetchall()
        return [r["ts"] for r in rows]


@pytest.fixture
def harness(mocker, tmp_path):
    return Harness(mocker, tmp_path)
