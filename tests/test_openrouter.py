import asyncio
import json

import httpx

from llmbot import openrouter
from llmbot.openrouter import OpenRouter

REPLY = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
         "provider": "Other", "usage": {"cost": 0.0001}}


async def test_hanging_provider_falls_back_to_default_routing(monkeypatch):
    monkeypatch.setattr(openrouter, "CHAT_DEADLINE", 0.05)
    payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if "provider" in payloads[-1]:  # the cheapest provider hangs
            await asyncio.sleep(1)
        return httpx.Response(200, json=REPLY)

    client = httpx.AsyncClient(base_url=openrouter.BASE_URL, transport=httpx.MockTransport(handler))
    result = await OpenRouter("k", client=client).chat(model="m", messages=[{"role": "user", "content": "x"}])
    assert result.text == "ok"
    assert [("provider" in p) for p in payloads] == [True, False]
    assert payloads[0]["reasoning"] == {"enabled": False}
