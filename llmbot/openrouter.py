"""Minimal async OpenRouter client: chat completions, image generation, audio input.

Only what the bot needs, on top of httpx, so one client covers the three
OpenRouter endpoints without three SDKs.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("llmbot.openrouter")

BASE_URL = "https://openrouter.ai/api/v1"
# OpenRouter keeps a slow request alive with keep-alive bytes, so httpx's read
# timeout never fires on a provider that hangs: these deadlines cover the whole call.
CHAT_DEADLINE = 60.0
IMAGE_DEADLINE = 150.0


class OpenRouterError(Exception):
    def __init__(self, message: str, status: int | str | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None
    model: str
    cost: float = 0.0
    truncated: bool = False
    # Raw assistant message (to append to the conversation) and its tool calls
    message: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class ImageResult:
    data: bytes
    media_type: str
    model: str
    cost: float = 0.0


@dataclass
class ModelInfo:
    id: str
    name: str
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)


def data_url(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"


class OpenRouter:
    def __init__(self, api_key: str, *, timeout: float = 180.0, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/emoubarak/signal-bot",
                "X-Title": "signal-llm-bot",
            },
        )
        self._models: dict[str, ModelInfo] | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, payload: dict, deadline: float) -> dict:
        try:
            resp = await asyncio.wait_for(self._client.post(path, json=payload), deadline)
        except TimeoutError as e:
            raise OpenRouterError(f"no answer after {deadline:.0f} s", "timeout") from e
        except httpx.HTTPError as e:
            raise OpenRouterError(f"network error: {e}") from e
        if resp.status_code >= 400:
            detail = resp.text[:300]
            try:
                detail = resp.json().get("error", {}).get("message", detail)
            except Exception:
                pass
            raise OpenRouterError(f"HTTP {resp.status_code}: {detail}", resp.status_code)
        body = resp.json()
        # OpenRouter can answer 200 with an error object (e.g. provider failure)
        if isinstance(body, dict) and body.get("error") and not body.get("choices") and not body.get("data"):
            err = body["error"]
            raise OpenRouterError(str(err.get("message", err)), err.get("code"))
        return body

    # ── chat ────────────────────────────────────────────────────────────────

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: list[dict] | None = None,
        retry_on_truncation: bool = True,
    ) -> ChatResult:
        """Chat completion, optionally with tools (OpenAI format). A reply cut by
        `max_tokens` is retried once with a doubled budget; if still cut,
        `truncated` is set so the caller can say so."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
            # cheapest provider that supports the request (tools, images); open models vary up to 10x
            "provider": {"sort": "price"},
            # a chat bot's replies gain little from reasoning and it makes them 2-3x slower
            "reasoning": {"enabled": False},
        }
        if tools:
            payload["tools"] = tools
        try:
            body = await self._post("/chat/completions", payload, CHAT_DEADLINE)
        except OpenRouterError as e:
            if e.status != "timeout":
                raise
            # the cheapest provider hangs: let OpenRouter route to any other one
            logger.warning("chat %s: %s, retrying with default routing", model, e)
            del payload["provider"]
            body = await self._post("/chat/completions", payload, CHAT_DEADLINE)
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        text = (content or "").strip()
        finish = choice.get("finish_reason")
        cost = float((body.get("usage") or {}).get("cost") or 0.0)
        result = ChatResult(text=text, finish_reason=finish, model=body.get("model", model), cost=cost,
                            message=message, tool_calls=list(message.get("tool_calls") or []))
        logger.info("chat %s (%s) finish=%s cost=$%.5f", result.model, body.get("provider"), finish, cost)

        if finish == "length":
            if retry_on_truncation:
                logger.info("reply truncated, retrying with max_tokens=%d", max_tokens * 2)
                second = await self.chat(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens * 2, tools=tools, retry_on_truncation=False,
                )
                second.cost += cost
                return second
            result.truncated = True
        return result

    async def transcribe(self, *, model: str, audio: bytes, audio_format: str, prompt: str) -> ChatResult:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode(), "format": audio_format}},
            ],
        }]
        return await self.chat(model=model, messages=messages, temperature=0.0, max_tokens=2000)

    # ── images ──────────────────────────────────────────────────────────────

    async def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        aspect_ratio: str = "1:1",
        references: list[tuple[bytes, str]] | None = None,
    ) -> ImageResult:
        """Text-to-image, or image-to-image when `references` are given."""
        payload: dict = {"model": model, "prompt": prompt, "aspect_ratio": aspect_ratio, "n": 1}
        if references:
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": data_url(data, media_type)}}
                for data, media_type in references
            ]
        body = await self._post("/images", payload, IMAGE_DEADLINE)
        items = body.get("data") or []
        if not items or not items[0].get("b64_json"):
            raise OpenRouterError("no image in response")
        cost = float((body.get("usage") or {}).get("cost") or 0.0)
        logger.info("image %s cost=$%.4f", model, cost)
        return ImageResult(
            data=base64.b64decode(items[0]["b64_json"]),
            media_type=items[0].get("media_type") or "image/png",
            model=model,
            cost=cost,
        )

    # ── models ──────────────────────────────────────────────────────────────

    async def models(self, refresh: bool = False) -> dict[str, ModelInfo]:
        if self._models is None or refresh:
            try:
                resp = await self._client.get("/models")
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise OpenRouterError(f"cannot list models: {e}") from e
            self._models = {}
            for m in resp.json().get("data", []):
                arch = m.get("architecture") or {}
                self._models[m["id"]] = ModelInfo(
                    id=m["id"], name=m.get("name", m["id"]),
                    input_modalities=arch.get("input_modalities") or [],
                    output_modalities=arch.get("output_modalities") or [],
                )
        return self._models
