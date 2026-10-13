"""A local OpenAI-compatible server (llama-swap, llama-server…) as the chat endpoint."""
import json

import httpx
import pytest

from llmbot import openrouter
from llmbot.agent import IMAGE_UNAVAILABLE, Agent, tools_for
from llmbot.config import Settings
from llmbot.openrouter import OpenRouter

REPLY = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
         "model": "qwen3.8-flash-next-q2_0"}
LOCAL_URL = "http://100.111.115.112:8089/v1"


def local_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=LOCAL_URL, transport=httpx.MockTransport(handler))


# ── client ──────────────────────────────────────────────────────────────────

async def test_local_chat_sends_no_openrouter_only_fields():
    """provider/reasoning/usage are OpenRouter extensions: a local server may reject them."""
    payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=REPLY)

    llm = OpenRouter("", base_url=LOCAL_URL, local=True, client=local_client(handler))
    result = await llm.chat(model="qwen3.8-flash-next-ablit", messages=[{"role": "user", "content": "x"}])

    assert result.text == "ok"
    assert result.model == "qwen3.8-flash-next-q2_0"
    assert "provider" not in payloads[0]
    assert "reasoning" not in payloads[0]
    assert "usage" not in payloads[0]


async def test_local_chat_keeps_tools():
    payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=REPLY)

    llm = OpenRouter("", base_url=LOCAL_URL, local=True, client=local_client(handler))
    tools = [{"type": "function", "function": {"name": "get_history", "parameters": {}}}]
    await llm.chat(model="m", messages=[{"role": "user", "content": "x"}], tools=tools)

    assert payloads[0]["tools"] == tools


async def test_local_chat_does_not_retry_without_provider(monkeypatch):
    """The 'drop provider and retry' path is OpenRouter routing: a local server has none."""
    monkeypatch.setattr(openrouter, "LOCAL_CHAT_DEADLINE", 0.05)
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, json={"error": {"message": "boom"}})

    llm = OpenRouter("", base_url=LOCAL_URL, local=True, client=local_client(handler))
    with pytest.raises(openrouter.OpenRouterError):
        await llm.chat(model="m", messages=[{"role": "user", "content": "x"}])
    assert len(calls) == 1


async def test_local_timeout_is_longer_than_openrouter():
    assert openrouter.LOCAL_CHAT_DEADLINE > openrouter.CHAT_DEADLINE


async def test_local_chat_retries_truncated_reply_once_with_a_bigger_budget():
    payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        finish = "length" if len(payloads) == 1 else "stop"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": finish}]})

    llm = OpenRouter("", base_url=LOCAL_URL, local=True, client=local_client(handler))
    await llm.chat(model="m", messages=[{"role": "user", "content": "x"}], max_tokens=100)

    assert [p["max_tokens"] for p in payloads] == [100, 200]


# ── capabilities ────────────────────────────────────────────────────────────

def settings(**overrides) -> Settings:
    base = {"signal_service": "127.0.0.1:8080", "phone_number": "+33700000000", "openrouter_api_key": ""}
    return Settings(**{**base, **overrides})


def test_no_key_but_a_local_endpoint_is_enough():
    s = Settings.from_env({"SIGNAL_SERVICE": "s", "PHONE_NUMBER": "p", "LOCAL_BASE_URL": LOCAL_URL + "/"})
    assert s.openrouter_api_key == ""
    assert s.local_chat
    assert s.chat_endpoint == LOCAL_URL  # trailing slash trimmed, or the paths would double up


def test_no_key_and_no_local_endpoint_is_refused():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        Settings.from_env({"SIGNAL_SERVICE": "s", "PHONE_NUMBER": "p"})


def test_chat_model_falls_back_to_the_older_variable_name():
    env = {"SIGNAL_SERVICE": "s", "PHONE_NUMBER": "p", "OPENROUTER_API_KEY": "k"}
    assert Settings.from_env({**env, "OPENROUTER_MODEL": "old/slug"}).chat_model == "old/slug"
    assert Settings.from_env({**env, "CHAT_MODEL": "new/slug", "OPENROUTER_MODEL": "old/slug"}).chat_model == "new/slug"


def test_empty_image_and_transcribe_models_disable_those_capabilities():
    s = Settings.from_env({"SIGNAL_SERVICE": "s", "PHONE_NUMBER": "p", "LOCAL_BASE_URL": LOCAL_URL,
                           "IMAGE_MODEL": "", "TRANSCRIBE_MODEL": ""})
    assert not s.images_enabled
    assert not s.transcription_enabled


def test_defaults_keep_images_and_transcription_on():
    s = Settings.from_env({"SIGNAL_SERVICE": "s", "PHONE_NUMBER": "p", "OPENROUTER_API_KEY": "k"})
    assert s.images_enabled and s.transcription_enabled


def test_image_tool_is_not_offered_when_images_are_off():
    names = [t["function"]["name"] for t in tools_for(settings(image_model=""))]
    assert names == ["get_history", "get_spend"]


def test_image_tool_is_offered_when_images_are_on():
    names = [t["function"]["name"] for t in tools_for(settings(image_model="meta/muse-image"))]
    assert names == ["get_history", "generate_image", "get_spend"]


async def test_image_tool_refuses_when_images_are_off():
    """The model must not promise an image it cannot make, even if it calls the tool anyway."""
    agent = Agent(settings(image_model=""), store=None, llm=None, registry=None, tz=None)
    result = await agent._run_tool("generate_image", {"prompt": "un chat"}, req=None, images={}, result=None)
    assert result == IMAGE_UNAVAILABLE
