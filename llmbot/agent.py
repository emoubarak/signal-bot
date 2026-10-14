"""Natural-language mode: the model reads the request, decides what to do and
calls tools (history, image generation, spend, model switch), then answers."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from llmbot import prompts
from llmbot.personas import Registry
from llmbot.config import Settings
from llmbot.context import format_message
from llmbot.media import image_for_model, image_for_signal
from llmbot.openrouter import OpenRouter, OpenRouterError, data_url
from llmbot.store import Store, StoredMessage

logger = logging.getLogger("llmbot.agent")

SILENCE = "[nothing]"
# There is no local equivalent of OpenRouter's Image API: without an image model
# the tool is not offered at all, so the model cannot promise an image it cannot make.
IMAGE_UNAVAILABLE = "Refused: image creation is disabled on this bot (empty IMAGE_MODEL in .env)."
ASPECT_RATIOS = {"1:1", "1:2", "2:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "9:21", "auto"}
MAX_ROUNDS = 5
MAX_TOOL_HISTORY = 300

GET_HISTORY_TOOL = {"type": "function", "function": {
    "name": "get_history",
    "description": "Reads more messages from this conversation than the recent ones provided. "
                   "Useful to summarize a long period or find something old.",
    "parameters": {"type": "object", "properties": {
        "count": {"type": "integer", "description": "Number of messages to read (max 300)."},
        "since_hours": {"type": "number", "description": "Only keep messages from the last N hours."},
    }},
}}
GENERATE_IMAGE_TOOL = {"type": "function", "function": {
    "name": "generate_image",
    "description": "Generates an image, or edits/combines conversation images passed as references.",
    "parameters": {"type": "object", "properties": {
        "prompt": {"type": "string", "description": "Precise description of the wanted image (or of the edit)."},
        "aspect_ratio": {"type": "string", "enum": sorted(ASPECT_RATIOS),
                         "description": "Format. 'auto' when starting from a reference image."},
        "reference_images": {"type": "array", "items": {"type": "string"},
                             "description": "Image identifiers from the conversation (img1, img2…) to use as references."},
    }, "required": ["prompt"]},
}}
GET_SPEND_TOOL = {"type": "function", "function": {
    "name": "get_spend",
    "description": "What the bot cost (today, 30 days).",
    "parameters": {"type": "object", "properties": {}},
}}


def tools_for(settings: Settings) -> list[dict]:
    """The tools the model is offered, given what this bot can actually do."""
    tools = [GET_HISTORY_TOOL]
    if settings.images_enabled:
        tools.append(GENERATE_IMAGE_TOOL)
    tools.append(GET_SPEND_TOOL)
    return tools


@dataclass
class AgentResult:
    text: str
    images: list[tuple[bytes, str]] = field(default_factory=list)
    image_costs: list[tuple[str, float]] = field(default_factory=list)  # (model, cost)
    chat_cost: float = 0.0
    chat_model: str = ""
    silent: bool = False


@dataclass
class AgentRequest:
    chat_id: str
    chat_name: str | None
    user_id: str
    thread: list[StoredMessage]      # reply chain ending with the current message
    window: list[StoredMessage]      # recent messages before the thread, for context
    quoted: StoredMessage | None     # the message replied to, if any (may be in thread)
    images_remaining: int
    # Timestamp of the last /clear in this chat: nothing before it is shown to the
    # model, and get_history cannot reach past it. None = never cleared.
    context_start: int | None = None


class Agent:
    def __init__(self, settings: Settings, store: Store, llm: OpenRouter, registry: Registry, tz: ZoneInfo) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.registry = registry
        self.tz = tz

    # ── prompt building ─────────────────────────────────────────────────────

    def _system(self, req: AgentRequest) -> str:
        s = self.settings
        now = datetime.now(self.tz)
        personas = ""
        if self.registry.personas:
            personas = "\n\nCharacters or styles you may be asked to adopt (follow the instruction):\n" + "\n".join(
                f"- {c.name} : {c.prompt.replace(chr(10), ' ')[:300]}" for c in self.registry.personas)
        where = f' in the group "{req.chat_name}"' if req.chat_name else " in private"
        can: list[str] = [
            '- summarize the conversation (a period, the last N messages, "since yesterday"): the latest messages '
            "are provided below, call get_history if you need more or older ones;",
            "- answer a question, explain, give an opinion, analyze an exchange, roast someone (biting "
            "joke between friends, never about looks, origin or religion), argue a debate pro/con, "
            "speak in a character's voice;",
        ]
        if s.images_enabled:
            can.append(
                "- create an image with generate_image, or edit/combine conversation images by passing "
                "their identifiers (img1, img2…) as references: a single generation per request unless "
                "several are explicitly asked for;")
        if s.transcription_enabled:
            can.append("- voice notes are already transcribed (🎤 lines): if asked what a voice note says, quote it;")
        else:
            can.append(
                "- voice notes are not transcribed here: if asked what a voice note says, say "
                "transcription is off and the content can be retyped as text;")
        if req.context_start is not None:
            can.append(
                "- the context was cleared in this conversation: you see nothing said before, and "
                "get_history cannot reach back past it either. If asked to summarize an earlier period, "
                "simply say so instead of inventing.")
        if not s.images_enabled:
            can.append(
                "- image creation and editing are off: if asked for one, simply say so, "
                "without inventing an image;")
        can.append("- get_spend for the cost. Your model, only the owner changes it: if asked to "
                   "change it, say it is not your call.")
        return (
            f"You are {s.bot_name}, a participant{where} on Signal. People address you by calling your name, "
            "by mentioning you, by replying to one of your messages or in private. You understand the request in "
            "natural language and act, without asking for confirmation when it is clear.\n"
            + prompts.PERSONALITY + "\n"
            "What you can do:\n"
            + "\n".join(can) + "\n\n"
            'If a message is marked [TARGET], that is what this is about ("summarize this", "roast him", "redo this '
            'image"). When spoken to directly, even just to greet or hail you '
            f'("yo {s.bot_name}", "hi {s.bot_name}", "{s.bot_name}?"), you answer. Reply exactly '
            + SILENCE + " only if people talk about you in the third person without addressing you "
            f'("{s.bot_name} crashed yesterday").\n'
            + ("When you generate an image, your final text is a short caption, or nothing.\n" if s.images_enabled else "")
            + prompts.COMMON_RULES.strip()
            + personas
            + f"\n\nNow: {now.strftime('%A %d/%m/%Y %H:%M')} ({s.timezone})."
            + (f" Remaining image allowance for this person: {req.images_remaining}." if s.images_enabled else "")
        )

    def _line(self, m: StoredMessage, images: dict[str, dict]) -> str:
        line = format_message(m, self.tz, self._reply_name)
        ids = []
        for img in m.images():
            if Path(img["path"]).exists():
                key = f"img{len(images) + 1}"
                images[key] = img
                ids.append(key)
        return line + (f" (image : {', '.join(ids)})" if ids else "")

    def _reply_name(self, m: StoredMessage) -> str | None:
        t = self.store.get(m.chat_id, m.reply_to_ts) if m.reply_to_ts else None
        return t.author_name if t else None

    def _build_messages(self, req: AgentRequest, images: dict[str, dict]) -> list[dict]:
        system = self._system(req)
        # After a /clear, nothing said before it may reach the model: drop it here too,
        # so a caller that hands over an older thread cannot undo the command.
        floor = req.context_start
        window = [m for m in req.window if floor is None or m.ts >= floor]
        thread = [m for m in req.thread if floor is None or m.ts >= floor] or req.thread[-1:]
        quoted = req.quoted if floor is None or (req.quoted and req.quoted.ts >= floor) else None
        if window:
            system += "\n\n[CONVERSATION - recent messages]\n" + "\n".join(self._line(m, images) for m in window)
        messages: list[dict] = [{"role": "system", "content": system}]

        quoted_in_thread = quoted is not None and any(m.ts == quoted.ts for m in thread)
        for i, m in enumerate(thread):
            if m.is_bot:
                messages.append({"role": "assistant", "content": m.text})
                continue
            text = self._line(m, images)
            is_last = i == len(thread) - 1
            if is_last and quoted is not None and not quoted_in_thread:
                text = "[TARGET]\n" + self._line(quoted, images) + "\n\n" + text
            content: list[dict] = [{"type": "text", "text": text}]
            for img in (m.images() + (quoted.images() if is_last and quoted and not quoted_in_thread else []))[-3:]:
                path = Path(img["path"])
                if path.exists():
                    data, mt = image_for_model(path.read_bytes(), img["content_type"])
                    content.append({"type": "image_url", "image_url": {"url": data_url(data, mt)}})
            messages.append({"role": "user", "content": content})
        return messages

    # ── tools ───────────────────────────────────────────────────────────────

    async def _run_tool(self, name: str, args: dict, req: AgentRequest, images: dict[str, dict],
                        result: AgentResult) -> str:
        if name == "get_history":
            count = max(1, min(int(args.get("count") or 100), MAX_TOOL_HISTORY))
            after = None
            if args.get("since_hours"):
                after = int((time.time() - float(args["since_hours"]) * 3600) * 1000)
            # a /clear forbids reaching past it, whatever the model asks for
            if req.context_start is not None:
                after = max(after or 0, req.context_start)
            before = req.thread[-1].ts
            msgs = self.store.recent(req.chat_id, count, before_ts=before, after_ts=after)
            if not msgs:
                if req.context_start is not None:
                    return ("Nothing since /clear: the context was cleared, I cannot see what was said before. "
                            "Tell me what I need to know, or ask me to summarize from now on.")
                return "No messages in this period."
            return "\n".join(self._line(m, images) for m in msgs)

        if name == "generate_image":
            if not self.settings.images_enabled:
                return IMAGE_UNAVAILABLE
            if result.images and len(result.images) >= max(1, req.images_remaining):
                return "Refused: image limit reached for this person."
            if req.images_remaining <= 0:
                return "Refused: image limit reached for this person."
            prompt = str(args.get("prompt") or "").strip()
            if not prompt:
                return "A prompt is required."
            refs: list[tuple[bytes, str]] = []
            unknown = []
            for key in args.get("reference_images") or []:
                img = images.get(str(key))
                if img and Path(img["path"]).exists():
                    refs.append(image_for_model(Path(img["path"]).read_bytes(), img["content_type"]))
                else:
                    unknown.append(str(key))
            if unknown:
                return f"Unknown images: {', '.join(unknown)}. Valid identifiers: {', '.join(images) or 'none'}."
            ratio = str(args.get("aspect_ratio") or ("auto" if refs else "1:1"))
            if ratio not in ASPECT_RATIOS:
                ratio = "auto" if refs else "1:1"
            try:
                out = await self.llm.generate_image(model=self.settings.image_model, prompt=prompt,
                                                    aspect_ratio=ratio, references=refs[:8] or None)
            except OpenRouterError as e:
                return f"Generation failed: {e}"
            result.images.append(image_for_signal(out.data, out.media_type))
            result.image_costs.append((out.model, out.cost))
            return "Image generated, it will be sent with your reply. Now reply with a short caption."

        if name == "get_spend":
            now = datetime.now(self.tz)
            day = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
            month = int((now - timedelta(days=30)).timestamp() * 1000)
            return (f"Today: {self.store.cost_since(day, req.chat_id):.3f} $ here, "
                    f"{self.store.cost_since(day):.3f} $ in total. 30 days: "
                    f"{self.store.cost_since(month, req.chat_id):.3f} $ here, {self.store.cost_since(month):.3f} $ in total.")

        return f"Unknown tool: {name}"

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self, req: AgentRequest, *, chat_model: str) -> AgentResult:
        images: dict[str, dict] = {}
        messages = self._build_messages(req, images)
        result = AgentResult(text="", chat_model=chat_model)
        tools = tools_for(self.settings)

        for _ in range(MAX_ROUNDS):
            reply = await self.llm.chat(model=chat_model, messages=messages, temperature=0.7,
                                        max_tokens=self.settings.max_output_tokens, tools=tools)
            result.chat_cost += reply.cost
            result.chat_model = reply.model
            if not reply.tool_calls:
                result.text = reply.text + (" […]" if reply.truncated else "")
                break
            messages.append(reply.message)
            for call in reply.tool_calls:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                logger.info("tool %s %s", fn.get("name"), json.dumps(args, ensure_ascii=False)[:200])
                output = await self._run_tool(str(fn.get("name")), args, req, images, result)
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": output})
        else:
            # ran out of rounds while still calling tools: ask for the final answer without tools
            reply = await self.llm.chat(model=chat_model, messages=messages, temperature=0.7,
                                        max_tokens=self.settings.max_output_tokens)
            result.chat_cost += reply.cost
            result.text = reply.text

        # the model sometimes copies the "[HH:MM] Name : " header of the transcript it was shown
        result.text = re.sub(rf"^\s*(\[\d{{1,2}}:\d{{2}}\]\s*)?{re.escape(self.settings.bot_name)}\s*:\s*", "",
                             result.text, flags=re.IGNORECASE) if result.text.lstrip().startswith("[") else result.text
        if result.text.strip() == SILENCE or (not result.text.strip() and not result.images):
            result.silent = not result.images
            result.text = ""
        return result
