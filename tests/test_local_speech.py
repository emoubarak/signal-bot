"""Voice notes on a local speech server (TRANSCRIBE_BASE_URL, NeMo Parakeet on minisforum, 2026-11-07),
OpenRouter only as a fallback."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from signalbot.test_utils.chat_testing import _new_timestamp

from llmbot.media import foreign_script, plan_cuts
from llmbot.openrouter import OpenRouterError
from llmbot.store import KIND_VOICE
from tests.conftest import CHAT_ID, PAUL, Harness, group_message, voice_attachment

STT = "http://127.0.0.1:8094/v1"
RETRY = "http://127.0.0.1:8093/v1"


@pytest.fixture
def make(mocker, tmp_path):
    mocker.patch("llmbot.handler.mp3_pieces", new_callable=AsyncMock, return_value=[b"mp3"])
    return lambda **kw: Harness(mocker, tmp_path, transcribe_base_url=STT, **kw)


async def _voice(h: Harness) -> str:
    t = _new_timestamp()
    await h.run(group_message(PAUL, None, attachments=[voice_attachment()], ts=t))
    stored = h.store.get(CHAT_ID, t)
    assert stored.kind == KIND_VOICE
    return stored.text


async def test_local_server_first(make):
    h = make()
    assert await _voice(h) == "bonjour mop ici une note vocale"
    assert [(c["base_url"], c["format"], c["audio"]) for c in h.llm.local_stt_calls] == [(STT, "mp3", b"mp3")]
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


def test_cuts_on_pauses_at_most_90_s():
    assert plan_cuts(60, []) == [(0.0, 60)]
    assert plan_cuts(300, []) == [(0.0, 90.0), (90.0, 180.0), (180.0, 270.0), (270.0, 300)]
    # the latest pause that fits wins; one before 30 s into the piece is not used
    assert plan_cuts(200, [10, 50, 80, 95, 160]) == [(0.0, 80), (80, 160), (160, 200)]


def test_foreign_script():
    assert foreign_script("Тест, тест, один, две")
    assert not foreign_script("Des tests, un, deux, ça marche ? 🤣")
    assert not foreign_script("Mop c'est Θ le meilleur")
    assert not foreign_script("123 !")


async def test_pieces_are_sent_in_order_and_joined(mocker, tmp_path):
    mocker.patch("llmbot.handler.mp3_pieces", new_callable=AsyncMock, return_value=[b"one", b"two"])
    h = Harness(mocker, tmp_path, transcribe_base_url=STT)
    h.llm.local_heard = "bout de phrase"
    assert await _voice(h) == "bout de phrase bout de phrase"
    assert [c["audio"] for c in h.llm.local_stt_calls] == [b"one", b"two"]


async def test_another_alphabet_is_heard_again_told_the_language(make):
    h = make(transcribe_retry_url=RETRY, transcribe_language="fr")
    h.llm.local_heard = {STT: "Тест, тест, один, две", RETRY: "Des tests un deux un deux"}
    assert await _voice(h) == "Des tests un deux un deux"
    assert h.llm.local_stt_calls[-1]["base_url"] == RETRY and h.llm.local_stt_calls[-1]["language"] == "fr"
    # the retry server fails: the first text is kept rather than nothing
    h.llm.local_heard = {STT: "Тест, тест", RETRY: OpenRouterError("down")}
    assert await _voice(h) == "Тест, тест"
