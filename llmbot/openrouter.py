"""Minimal async OpenAI-compatible client: chat completions, image generation, audio input.

Only what the bot needs, on top of httpx, so one client covers the three
endpoints without three SDKs. Chat can be served either by OpenRouter or by a
local OpenAI-compatible server (llama-swap, llama-server, vLLM, Ollama); images
and voice notes stay on OpenRouter, the only one of the two that offers them.
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
# A local model runs on someone's own machine: a big mixture takes seconds to read
# a long context, and llama-swap may have to load the model before answering.
LOCAL_CHAT_DEADLINE = 300.0


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
    """OpenRouter for everything, or a local server for chat and OpenRouter for the rest.

    `base_url` and `api_key` are the chat endpoint: OpenRouter by default, or a
    local OpenAI-compatible server (no key) when the bot is configured with
    LOCAL_BASE_URL. Image generation and audio input always go to OpenRouter.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        local: bool = False,
        timeout: float = 180.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._timeout = timeout
        # clients replaced by reconfigure(): requests in flight may still use them, closed with the bot
        self._retired: list[httpx.AsyncClient] = []
        self._models: dict[str, ModelInfo] | None = None
        self._build(api_key, base_url, local, client)

    def _build(self, api_key: str, base_url: str, local: bool, client: httpx.AsyncClient | None = None) -> None:
        self.local = local
        headers = {"X-Title": "signal-llm-bot"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if not local:
            headers["HTTP-Referer"] = "https://github.com/emoubarak/signal-bot"
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=self._timeout, headers=headers)
        # Images and voice notes are OpenRouter-only: a second client, only built
        # when chat is local and the key is set.
        self._openrouter: httpx.AsyncClient | None = None
        if local and api_key:
            self._openrouter = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/emoubarak/signal-bot",
                    "X-Title": "signal-llm-bot",
                },
            )

    def reconfigure(self, api_key: str, *, base_url: str = BASE_URL, local: bool = False) -> None:
        """Switch provider or key while the bot runs (admin page). New requests use the new clients;
        the old ones are kept until close() so a request in flight is not cut."""
        self._retired.extend(c for c in (self._client, self._openrouter) if c is not None)
        self._build(api_key, base_url, local)
        self._models = None

    @property
    def _media_client(self) -> httpx.AsyncClient:
        """The client for OpenRouter-only endpoints (images, audio input)."""
        if self.local and self._openrouter is None:
            # chat is local and no key is set: these two endpoints were switched off
            # in the settings, so reaching here is a configuration mistake
            raise OpenRouterError("images and voice notes need OPENROUTER_API_KEY "
                                  "(a local chat server serves neither)", "no_key")
        return self._openrouter or self._client

    async def close(self) -> None:
        await self._client.aclose()
        if self._openrouter is not None:
            await self._openrouter.aclose()
        for old in self._retired:
            await old.aclose()

    async def _post(self, path: str, payload: dict, deadline: float,
                    client: httpx.AsyncClient | None = None) -> dict:
        client = client or self._client
        try:
            resp = await asyncio.wait_for(client.post(path, json=payload), deadline)
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
        try:
            body = resp.json()
        except ValueError as e:
            raise OpenRouterError(f"invalid JSON from {resp.request.url}: {resp.text[:200]}") from e
        # OpenRouter can answer 200 with an error object (e.g. provider failure)
        if isinstance(body, dict) and body.get("error") and not body.get("choices") and not body.get("data"):
            err = body["error"]
            if isinstance(err, dict):
                raise OpenRouterError(str(err.get("message", err)), err.get("code"))
            raise OpenRouterError(str(err))
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
        }
        if not self.local:
            payload["usage"] = {"include": True}
            # cheapest provider that supports the request (tools, images); open models vary up to 10x
            payload["provider"] = {"sort": "price"}
            # a chat bot's replies gain little from reasoning and it makes them 2-3x slower
            payload["reasoning"] = {"enabled": False}
        else:
            # same reason, for a local Qwen-style template (llama.cpp, vLLM): 7-14 s down to 2-4 s per answer
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            payload["tools"] = tools
        deadline = LOCAL_CHAT_DEADLINE if self.local else CHAT_DEADLINE
        try:
            body = await self._post("/chat/completions", payload, deadline)
        except OpenRouterError as e:
            # the cheapest provider hangs: let OpenRouter route to any other one.
            # A local server has only itself to fall back to.
            if e.status != "timeout" or self.local:
                raise
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
        logger.info("chat %s (%s) finish=%s cost=$%.5f", result.model,
                    body.get("provider") or ("local" if self.local else "?"), finish, cost)

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
        """Voice note to text. Audio input is an OpenRouter (Gemini-class) feature:
        a local text model ignores the audio part, so this always goes to OpenRouter."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode(), "format": audio_format}},
            ],
        }]
        payload = {
            "model": model, "messages": messages, "temperature": 0.0, "max_tokens": 2000,
            "usage": {"include": True},
        }
        body = await self._post("/chat/completions", payload, CHAT_DEADLINE, client=self._media_client)
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        cost = float((body.get("usage") or {}).get("cost") or 0.0)
        model_name = body.get("model", model)
        logger.info("transcribe %s cost=$%.5f", model_name, cost)
        return ChatResult(text=(content or "").strip(), finish_reason=choice.get("finish_reason"),
                          model=model_name, cost=cost, message=message)

    # ── images ──────────────────────────────────────────────────────────────

    async def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        aspect_ratio: str = "1:1",
        references: list[tuple[bytes, str]] | None = None,
    ) -> ImageResult:
        """Text-to-image, or image-to-image when `references` are given.
        OpenRouter only: a local chat server serves no /images endpoint."""
        payload: dict = {"model": model, "prompt": prompt, "aspect_ratio": aspect_ratio, "n": 1}
        if references:
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": data_url(data, media_type)}}
                for data, media_type in references
            ]
        body = await self._post("/images", payload, IMAGE_DEADLINE, client=self._media_client)
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
