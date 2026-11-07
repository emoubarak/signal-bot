"""Voice notes on a local speech server (TRANSCRIBE_BASE_URL, NeMo Parakeet on minisforum, 2026-11-07),
OpenRouter only as a fallback."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from signalbot.test_utils.chat_testing import _new_timestamp

from llmbot.openrouter import OpenRouterError
from llmbot.store import KIND_VOICE
from tests.conftest import CHAT_ID, PAUL, Harness, group_message, voice_attachment

STT = "http://127.0.0.1:8094/v1"


@pytest.fixture
def make(mocker, tmp_path):
    mocker.patch("llmbot.handler.to_mp3", new_callable=AsyncMock, return_value=b"mp3")
    return lambda **kw: Harness(mocker, tmp_path, transcribe_base_url=STT, **kw)


async def _voice(h: Harness) -> str:
    t = _new_timestamp()
    await h.run(group_message(PAUL, None, attachments=[voice_attachment()], ts=t))
    stored = h.store.get(CHAT_ID, t)
    assert stored.kind == KIND_VOICE
    return stored.text


async def test_local_server_first_with_the_bot_name_boosted(make):
    h = make()
    assert await _voice(h) == "bonjour mop ici une note vocale"
    assert h.llm.local_stt_calls == [{"base_url": STT, "format": "mp3", "phrases": ["Jarvis"]}]
    assert h.llm.transcribe_calls == []


async def test_openrouter_when_the_local_server_fails_or_hears_nothing(make):
    h = make()
    h.llm.local_heard = OpenRouterError("speech server: connection refused")
    assert await _voice(h) == "hi this is a voice note"
    h.llm.local_heard = ""
    assert await _voice(h) == "hi this is a voice note"
    assert len(h.llm.transcribe_calls) == 2


async def test_without_openrouter_model_the_failure_is_said(make):
    h = make(transcribe_model="")
    h.llm.local_heard = OpenRouterError("speech server: connection refused")
    assert await _voice(h) == "[voice note not transcribed: OpenRouterError]"
    h.llm.local_heard = ""
    assert await _voice(h) == "[inaudible]"
    assert h.llm.transcribe_calls == []
