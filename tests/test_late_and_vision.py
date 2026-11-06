"""Messages delivered after the bot was off, and photos described by a local model."""
from __future__ import annotations

from llmbot.handler import MAX_DESCRIBED
from llmbot.store import KIND_IMAGE
from tests.conftest import CHAT_ID, LEA, PAUL, Harness, group_message, image_attachment, now_ms


async def test_late_message_is_stored_but_not_answered(harness: Harness):
    two_hours_ago = now_ms() - 2 * 3_600_000
    await harness.run(group_message(PAUL, "jarvis are you there?", server_ts=two_hours_ago))
    assert harness.sent() == [] and harness.llm.chat_calls == []
    assert any("are you there" in m.text for m in harness.store.recent(CHAT_ID, 10))
    # once the backlog is in, a live call is answered and sees the late message
    await harness.run(group_message(LEA, "jarvis what did Paul ask?"))
    assert harness.sent_texts() == ["model reply"]
    assert "Paul: jarvis are you there?" in harness.llm.last_system()


async def test_lateness_uses_server_time_not_the_phone_clock(harness: Harness):
    # a phone set years back: its own timestamp is old, the server got it just now
    await harness.run(group_message(PAUL, "jarvis hi", ts=1_600_000_000_000))
    assert harness.sent_texts() == ["model reply"]


async def test_late_reply_minutes_zero_answers_everything(mocker, tmp_path):
    h = Harness(mocker, tmp_path, late_reply_minutes=0)
    await h.run(group_message(PAUL, "jarvis hello", server_ts=now_ms() - 86_400_000))
    assert h.sent_texts() == ["model reply"]


class LocalLLM:
    """FakeLLM answers, plus what a local llama-swap says it holds."""

    def __init__(self, base, loaded: bool) -> None:
        self._base, self.loaded, self.loaded_calls = base, loaded, []

    def __getattr__(self, name):
        return getattr(self._base, name)

    async def model_loaded(self, model: str) -> bool:
        self.loaded_calls.append(model)
        return self.loaded


def _local(mocker, tmp_path, loaded: bool) -> tuple[Harness, LocalLLM]:
    h = Harness(mocker, tmp_path, describe_images=True, local_base_url="http://127.0.0.1:8089/v1",
                openrouter_api_key="", chat_model="qwen3.8-flash-next-ablit")
    llm = LocalLLM(h.llm, loaded)
    h.handler.llm = llm
    return h, llm


async def test_local_model_describes_each_photo_of_an_album(mocker, tmp_path):
    h, llm = _local(mocker, tmp_path, loaded=True)
    h.llm.answer = "A cat on a sofa."
    album = [image_attachment() | {"id": f"pic{i}.png"} for i in range(MAX_DESCRIBED + 2)]
    await h.run(group_message(PAUL, "look", attachments=album))
    rec = h.store.recent(CHAT_ID, 1)[0]
    assert rec.kind == KIND_IMAGE
    assert rec.text.count("[image: A cat on a sofa.]") == MAX_DESCRIBED
    assert len(h.llm.chat_calls) == MAX_DESCRIBED and h.sent() == []
    assert h.llm.chat_calls[0]["model"] == "qwen3.8-flash-next-ablit"


async def test_photo_not_described_when_the_local_model_is_not_loaded(mocker, tmp_path):
    h, llm = _local(mocker, tmp_path, loaded=False)
    await h.run(group_message(PAUL, "look", attachments=[image_attachment()]))
    assert h.llm.chat_calls == [] and llm.loaded_calls == ["qwen3.8-flash-next-ablit"]
    assert "[image:" not in h.store.recent(CHAT_ID, 1)[0].text


async def test_model_loaded_reads_llama_swap_running(mocker):
    import httpx
    from llmbot.openrouter import OpenRouter

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/running"
        return httpx.Response(200, json={"running": [
            {"model": "qwen3.8-flash-next", "state": "ready"}, {"model": "embeddinggemma-2", "state": "ready"}]})

    client = httpx.AsyncClient(base_url="http://127.0.0.1:8089/v1", transport=httpx.MockTransport(handler))
    llm = OpenRouter("", base_url="http://127.0.0.1:8089/v1", local=True, client=client)
    assert await llm.model_loaded("qwen3.8-flash-next-ablit")  # selector, target loaded
    assert await llm.model_loaded("qwen3.8-flash-next")
    assert not await llm.model_loaded("qwen3.6-35b-ablit")

    down = httpx.AsyncClient(base_url="http://127.0.0.1:8089/v1",
                             transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    assert await OpenRouter("", base_url="http://127.0.0.1:8089/v1", local=True, client=down).model_loaded("x")


async def test_late_clear_from_the_owner_still_applies(mocker, tmp_path):
    from tests.conftest import PAUL
    h = Harness(mocker, tmp_path, owner_number=PAUL[0])
    await h.run(group_message(LEA, "old talk"))
    late = now_ms() - 3_600_000
    await h.run(group_message(PAUL, "/clear", server_ts=late))
    assert h.store.context_start(CHAT_ID) is not None
    assert any("starting fresh" in t for t in h.sent_texts())


async def test_cut_description_is_dropped_and_caption_is_given(mocker, tmp_path):
    from llmbot.openrouter import ChatResult
    h, llm = _local(mocker, tmp_path, loaded=True)
    seen = []

    async def cut(**kw):
        seen.append(kw["messages"][0]["content"][0]["text"])
        return ChatResult(text="A man next to a red car with a sign that says", finish_reason="length",
                          model="m", cost=0, truncated=True)
    h.llm.chat = cut
    await h.run(group_message(PAUL, "rate my face", attachments=[image_attachment()]))
    assert "[image:" not in h.store.recent(CHAT_ID, 1)[0].text
    assert "rate my face" in seen[0]


async def test_public_only_tries_the_next_address_when_one_is_dead(mocker):
    import asyncio
    import socket

    import httpcore

    from llmbot.webtools import _PublicOnly

    loop = asyncio.get_running_loop()

    async def two(host, port, *a, **k):
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800:21f:cb07:6820:80da:af6b:8b2c", port, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", port))]

    mocker.patch.object(loop, "getaddrinfo", side_effect=two)
    tried = []

    class Inner(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, **kw):
            tried.append(host)
            if ":" in host:
                raise httpcore.ConnectError("dead v6")
            return "stream"

    assert await _PublicOnly(Inner()).connect_tcp("example.com", 443) == "stream"
    assert tried == ["2606:2800:21f:cb07:6820:80da:af6b:8b2c", "93.184.215.14"]
