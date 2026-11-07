"""Natural-language mode: the model reads the request, decides what to do and
calls tools (history, image generation, spend, web search, weather…), then answers."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from llmbot import prompts, reminders, webtools
from llmbot.personas import Registry
from llmbot.config import Settings
from llmbot.context import fmt_time, format_message
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
# Said in the chat when the model's last word is still a tool call written as text.
NO_FINAL_ANSWER = "I got lost in my tools on that one, ask me again."
# A tool call the model wrote as text instead of a real call: Qwen's XML form
# (<tool_call><function=name><parameter=k>v</parameter></function></tool_call>) or JSON in <tool_call>.
_TOOL_MARKUP = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)|<function=[^>]*>.*?(?:</function>|\Z)", re.S)
_XML_CALL = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|\Z)", re.S)
_XML_PARAM = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.S)


def strip_tool_markup(text: str) -> str:
    """The text without any tool call written in it, so markup never reaches the chat."""
    return _TOOL_MARKUP.sub("", text or "").strip()


# 2026-11-06, local Qwen at Q2: one answer was the asker's line copied with its "[06:38] Name: " header,
# the next one the model's reasoning written as the answer ("I need to… The user is saying… Let me draft").
# Replayed on the 103 bot answers stored then: these two were the only ones flagged.
_LINE_HEADER = re.compile(r"^\s*\[(?:[A-Z][a-z]{2} |\d{2}/\d{2} )?\d{1,2}:\d{2}\]\s*([^:\n]{1,60}?)\s*:")
_REASONING = re.compile(
    r"\bthe user(?:'s)?\b|\blet me (?:draft|write|think|look|check|craft|answer|respond|reply)\b"
    r"|\b(?:draft|final answer|my reply|my response)\s*:"
    r"|^(?:okay|ok|alright|hmm)[,.]?\s+(?:so\s+)?(?:i|the|let|we)\b", re.I | re.M)
_PLANNING_OPENER = re.compile(r"^\s*(?:i need to|i should|i'll|i will|i must|we need to|let me|let's see)\b", re.I)
# Said to the model, never posted, when its answer is rejected; it gets one more try.
RETRY_NOTE = ("(Not posted: your last output {why}. Write only the message to post in the chat, in the language "
              "of the conversation: no reasoning, no notes about the request, no copy of a message.)")
_WHY = {"echo": "copied a message of the conversation instead of answering it",
        "reasoning": "was your reasoning about how to answer, not the answer"}
# Said to the model, never posted, when its answer is far above the word budget: it rewrites it shorter.
SHORTEN_NOTE = ("(Not posted: {words} words, the limit is {budget}. Rewrite it in at most {budget} words, same "
                "language and tone, keep the best of it. Only the new message, nothing else.)")
# The live conversation shown to the model: since the last pause this long, at most this many lines.
GAP_MS = 2 * 3600 * 1000
WINDOW_LINES = 12
SHOWN_GAP_MS = 30 * 60 * 1000   # a pause this long inside the window is written as "(40 min later)"
THREAD_TURNS = 7                # the end of the reply chain with the bot: at most 3 of its own replies


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def answer_problem(text: str, bot_name: str, recent: list[str], names: tuple[str, ...] = ()) -> str | None:
    """Why an answer must not be posted: "echo" (a line of the transcript copied back, header or not),
    "reasoning" (the model thinking aloud), or None when it is a reply. `names`: the people of the
    conversation, whose "Name:" in front of a copied message is still an echo."""
    if not text.strip():
        return None
    header = _LINE_HEADER.match(text)
    if header and header.group(1).strip().lower() != bot_name.lower():
        return "echo"
    # "Name: …" alone is a legit answer too (a profile of each person): an echo only when what follows is a
    # message of the conversation copied back (2026-11-07, a profiling answer was silenced by the plain rule)
    body = _LINE_HEADER.sub("", text, count=1).strip()
    for n in names:
        if n and n.lower() != bot_name.lower() and body.lower().startswith(n.lower() + ":"):
            body = body[len(n) + 1:]
            break
    body = _norm(body)
    if len(body) >= 20 and any(r and (body == _norm(r) or body.startswith(_norm(r)) and len(_norm(r)) >= 20)
                               for r in recent):
        return "echo"
    if _REASONING.search(text) or (_PLANNING_OPENER.match(text) and "\n\n" in text.strip()):
        return "reasoning"
    return None


# Reply length, decided in code (2026-11-07). Measured on the group: a person's turn is 9 words (median, p90 28),
# the bot's replies were 75 (p90 209) although the prompt said "same length as the others". Research agrees that
# vague length instructions do not hold and that models cannot count their own words: the budget is a number
# given with the request, counted afterwards, with one shorter rewrite when it is clearly exceeded.
BUDGETS = {"short": 30, "medium": 60, "long": 150}
# max_tokens per class: a safety net above the budget, wide enough for a tool call (a meme's captions)
TOKEN_CAPS = {"short": 300, "medium": 450, "long": 900}
OVER = 1.5   # words above budget × OVER: the model is asked once for a shorter version
CUT = 2.0    # still above budget × CUT after that: cut after the last whole sentence that fits
_LONG_ASK = re.compile(
    r"\br[ée]sum|\br[ée]cap|\bsummar|\bexpliqu|\bexplain|\blistes?\b|\blist\b|\bprofiling\b|\bprofils? de\b"
    r"|d[ée]taill|\banalys|\bcompar|\bconseils\b|\badvice\b|\brecettes?\b|\brecipes?\b|\b[ée]tapes\b|\bsteps\b"
    r"|\btuto(?:s|riel)?\b|\btradui|\btranslat", re.I)
_MEDIUM_ASK = re.compile(
    r"\broast|\bclash|\bblagues?\b|\bjokes?\b|\bimite|\bimitate|\bparle[rsz]?\b.{0,15}\bcomme\b|\btalk like"
    r"|\bpo[èe]mes?\b|\bpoems?\b|\bchansons?\b|\bsongs?\b|\bhistoire\b|\bstory\b|\bavis\b|\bopinion"
    r"|\bt.?en penses|\bpenses?[- ]tu\b|\btu penses\b|\btu crois\b|\bcrois[- ]tu\b|\bpourquoi\b|\bwhy\b"
    r"|\bcomment\b|\bhow\b|\bc.?est quoi\b|\bwhat is\b|\bc.?est qui\b|\bwho is\b|\bd[ée]bat|\bdebate"
    r"|\bargumente|\bd[ée]fends?\b|\bvs\b|\bou (?:bien|plut)|\bjeux? de mots?\b|\bpuns?\b|\bid[ée]es\b"
    r"|\bideas\b|\bsurnoms\b|\bnicknames\b|\bpunchlines?\b", re.I)
_SENTENCE_END = re.compile(r"[.!?…](?=\s|$)|\n")
# tools whose answer is a summary or a read: the reply may be long; a web lookup: at least medium
_LONG_TOOLS = {"get_history", "search_messages", "read_page"}


def length_class(text: str, *, with_image: bool = False) -> str:
    """How long the reply to `text` may be: short by default, long for a summary, an explanation, a list;
    medium for a roast, a joke, a character's voice, an opinion, a question why or how. A link read with
    read_page makes it long afterwards (class_after_tools), a link merely shared does not."""
    if _LONG_ASK.search(text):
        return "long"
    if _MEDIUM_ASK.search(text) or with_image:
        return "medium"
    return "short"


def class_after_tools(cls: str, used: set[str]) -> str:
    if used & _LONG_TOOLS:
        return "long"
    if "web_search" in used and cls == "short":
        return "medium"
    return cls


def word_count(text: str) -> int:
    return len(text.split())


def cut_to_words(text: str, limit: int, *, truncated: bool = False) -> str:
    """`text` ended after its last whole sentence within `limit` words. A reply cut off by max_tokens is
    brought back to its last whole sentence even when short. With no sentence end that fits: the words
    that fit and "…"."""
    if word_count(text) <= limit and not truncated:
        return text
    best = None
    for m in _SENTENCE_END.finditer(text):
        if word_count(text[:m.end()]) > limit:
            break
        if text[:m.end()].strip():
            best = m.end()
    if best is not None:
        return text[:best].strip()
    words = text.split()[:limit]
    return " ".join(words).rstrip(" ,;:") + "…"


def _pause(ms: int) -> str:
    minutes = ms // 60000
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h" + (f" {minutes % 60:02d}" if minutes % 60 else "")


def tool_calls_in_text(text: str) -> list[dict]:
    """Tool calls the model wrote as text, in the OpenAI shape, so they still run when the
    server did not recognise them. Parameter values are read as JSON when they are JSON
    (numbers, lists), as plain text otherwise."""
    calls = []
    for i, block in enumerate(re.findall(r"<tool_call>(.*?)(?:</tool_call>|\Z)", text or "", re.S)):
        name, args = None, {}
        m = _XML_CALL.search(block)
        if m:
            name = m.group(1)
            for key, value in _XML_PARAM.findall(m.group(2)):
                try:
                    args[key] = json.loads(value)
                except ValueError:
                    args[key] = value
        else:
            try:
                data = json.loads(block.strip())
                name, args = data.get("name"), data.get("arguments") or {}
            except (ValueError, AttributeError):
                continue
        if name:
            calls.append({"id": f"text_call_{i}", "type": "function",
                          "function": {"name": str(name), "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls
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
SEARCH_MESSAGES_TOOL = {"type": "function", "function": {
    "name": "search_messages",
    "description": "Finds old messages of this conversation by keywords (all the words must appear; accents and "
                   "capitals do not matter). Use it to find who said what, when, about a topic. get_history only "
                   "reads the latest messages.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Words to look for, e.g. 'restaurant lyon'."},
        "limit": {"type": "integer", "description": "How many matches to return (default 15, max 50)."},
        "since_hours": {"type": "number", "description": "Only look at the last N hours."},
    }, "required": ["query"]},
}}
REMIND_TOOL = {"type": "function", "function": {
    "name": "remind",
    "description": "Posts a message in this conversation later. Work out the time yourself from the current date "
                   "and time you were given.",
    "parameters": {"type": "object", "properties": {
        "message": {"type": "string", "description": "Exactly what to post when the time comes, in the language of "
                                                     "the conversation, addressed to whoever asked."},
        "in_minutes": {"type": "number", "description": "Post it this many minutes from now (for 'in 20 minutes')."},
        "at": {"type": "string", "description": "Or the moment itself, group time: YYYY-MM-DD HH:MM."},
    }, "required": ["message"]},
}}
LIST_REMINDERS_TOOL = {"type": "function", "function": {
    "name": "list_reminders",
    "description": "Lists the reminders still waiting in this conversation, with their numbers. Pass cancel_id to cancel one.",
    "parameters": {"type": "object", "properties": {
        "cancel_id": {"type": "integer", "description": "Number of the reminder to cancel."},
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


# What the prompt says about each web tool, in the order they are listed.
WEB_TOOL_HINTS = [
    ("web_search", "web_search for anything recent or that you do not know for sure (never guess a price, score or news)"),
    ("read_page", "read_page to read a link"),
    ("get_weather", "get_weather"),
    ("get_time", "get_time for another timezone"),
    ("calculate", "calculate for any arithmetic"),
    ("pick_random", "pick_random for heads or tails, dice and drawing between options (you cannot be random yourself)"),
]


def tools_for(settings: Settings) -> list[dict]:
    """The tools the model is offered, given what this bot can actually do."""
    tools = [GET_HISTORY_TOOL]
    if settings.tool_enabled("search_messages"):
        tools.append(SEARCH_MESSAGES_TOOL)
    if settings.tool_enabled("remind"):
        tools.extend([REMIND_TOOL, LIST_REMINDERS_TOOL])
    if settings.images_enabled:
        tools.append(GENERATE_IMAGE_TOOL)
    if settings.tool_enabled("get_spend"):
        tools.append(GET_SPEND_TOOL)
    if settings.web_tools:
        tools.extend(t for t in webtools.TOOLS if settings.tool_enabled(t["function"]["name"]))
    if settings.gifs_enabled:
        tools.append(webtools.SEND_GIF_TOOL)
    if settings.memes_enabled:
        tools.append(webtools.MAKE_MEME_TOOL)
    return tools


@dataclass
class AgentResult:
    text: str
    images: list[tuple[bytes, str]] = field(default_factory=list)
    memes: int = 0
    recycle_refusals: int = 0  # make_meme captions refused for recycling the bot's own jokes
    meme_subject: str = ""
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
            "- answer a question, explain, give an opinion, analyze an exchange, roast someone (in the tone "
            "the group uses between friends, no apology and no lecture), argue a debate pro/con, "
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
        if s.tool_enabled("search_messages"):
            can.append("- search_messages to find what someone said earlier by keywords (get_history only reads the "
                       "latest messages);")
        if s.tool_enabled("remind"):
            can.append("- remind to post a message in this chat later (\"remind us at 8pm\", \"in 20 minutes\"), "
                       "list_reminders to see or cancel them; work the time out from the date and time you are given;")
        if s.web_tools:
            lookups = [text for name, text in WEB_TOOL_HINTS if s.tool_enabled(name)]
            if lookups:
                can.append("- look things up: " + ", ".join(lookups) +
                           ". If a tool fails, say you could not look it up;")
        if s.gifs_enabled:
            can.append("- send_gif to react with an animated GIF when it says it better than words (rarely, "
                       "never for a serious subject; a single one per reply);")
        if s.memes_enabled:
            can.append("- make_meme to answer with a meme: a classic template, the subject of the request copied "
                       "from the request itself (never a name, an object or a joke picked up earlier), and the "
                       "captions about that subject in the language of the request; one per reply, when someone "
                       "asks for one or the joke lands better as a meme;")
        can.append(("- get_spend for the cost. " if s.tool_enabled("get_spend") else "- ") +
                   "Your model, only the owner changes it: if asked to change it, say it is not your call.")
        rules = Agent._tool_rules(s)
        return (
            f"You are {s.bot_name}, a participant{where} on Signal. People address you by calling your name, "
            "by mentioning you, by replying to one of your messages or in private. You understand the request in "
            "natural language and you do exactly what is asked, without asking for confirmation when it is clear.\n"
            + prompts.PERSONALITY + "\n"
            "What you can do:\n"
            + "\n".join(can) + "\n\n"
            + (("Tool rules, always followed:\n" + "\n".join(rules) + "\n\n") if rules else "")
            + 'If a message is marked [TARGET], that is what this is about ("summarize this", "roast him", "redo this '
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

    @staticmethod
    def _tool_rules(s) -> list[str]:
        """Imperatives for the cases a small model answers from memory instead of calling the tool (measured on
        Qwen3.5 4B: 11 of 18 tool scenes triggered without these, 15 with). Only for tools that are switched on."""
        rules = []
        if s.web_tools and s.tool_enabled("web_search"):
            rules.append("- A question about anything current or that you cannot know for certain (a ranking, a score, a "
                         "price, news, the latest version of something): call web_search FIRST, then answer from what it "
                         "returned. Never answer such a question from memory.")
        if s.web_tools and s.tool_enabled("calculate"):
            rules.append("- Any arithmetic, even simple (splitting a bill, a discount): call calculate.")
        recall = "call get_history" + ("; to find a past message by topic: call search_messages"
                                       if s.tool_enabled("search_messages") else "")
        rules.append(f"- A request to recap, summarize or recall what was said: {recall}.")
        if s.web_tools and s.tool_enabled("read_page"):
            rules.append("- A link in the conversation to read or summarize: call read_page.")
        return rules

    def _line(self, m: StoredMessage, images: dict[str, dict], with_time: bool = True) -> str:
        line = format_message(m, self.tz, self._reply_name, with_time=with_time)
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

    def _own(self, m: StoredMessage) -> bool:
        """A message the bot sent: text (KIND_BOT) or a meme, GIF or image (KIND_IMAGE, told by the author)."""
        return m.is_bot or m.author_id == self.settings.phone_number

    def _window(self, req: AgentRequest, thread: list[StoredMessage], quoted: StoredMessage | None,
                floor: int | None) -> list[StoredMessage]:
        """The people's messages since the last long pause, newest WINDOW_LINES. Not the bot's own earlier
        replies: they were 74 % of the words shown (2026-11-07), so the model copied its own length and
        running jokes instead of the group. Its replies in the current exchange are in `thread`."""
        skip = {m.ts for m in thread} | ({quoted.ts} if quoted else set())
        people = [m for m in req.window
                  if not self._own(m) and m.ts not in skip and (floor is None or m.ts >= floor)]
        kept: list[StoredMessage] = []
        after = thread[0].ts if thread else None
        for m in reversed(people):
            if after is not None and after - m.ts > GAP_MS:
                break
            kept.append(m)
            after = m.ts
            if len(kept) >= WINDOW_LINES:
                break
        return kept[::-1]

    def _build_messages(self, req: AgentRequest, images: dict[str, dict], budget: int | None = None) -> list[dict]:
        system = self._system(req)
        # After a /clear, nothing said before it may reach the model: drop it here too,
        # so a caller that hands over an older thread cannot undo the command.
        floor = req.context_start
        thread = [m for m in req.thread if floor is None or m.ts >= floor] or req.thread[-1:]
        thread = thread[-THREAD_TURNS:]
        while len(thread) > 1 and self._own(thread[0]):   # the exchange opens with a person, never with the bot
            thread = thread[1:]
        quoted = req.quoted if floor is None or (req.quoted and req.quoted.ts >= floor) else None
        window = self._window(req, thread, quoted, floor)
        if window:
            lines, prev = [], None
            for m in window:
                if prev is not None and m.ts - prev >= SHOWN_GAP_MS:
                    lines.append(f"({_pause(m.ts - prev)} later)")
                lines.append(self._line(m, images, with_time=False))
                prev = m.ts
            system += (f"\n\n[CONVERSATION - what people said lately, from {fmt_time(window[0].ts, self.tz)}; "
                       "your own earlier replies are not shown]\n" + "\n".join(lines))
        messages: list[dict] = [{"role": "system", "content": system}]

        quoted_in_thread = quoted is not None and any(m.ts == quoted.ts for m in thread)
        for i, m in enumerate(thread):
            if self._own(m):
                messages.append({"role": "assistant", "content": m.text})
                continue
            text = self._line(m, images, with_time=False)
            is_last = i == len(thread) - 1
            if is_last and quoted is not None and not quoted_in_thread:
                text = "[TARGET]\n" + self._line(quoted, images, with_time=False) + "\n\n" + text
            if is_last and budget:
                text += f"\n\n(Your reply: at most {budget} words.)"
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
        if name != "get_history" and not self.settings.tool_enabled(name):
            return "This tool is switched off."
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

        if name == "send_gif":
            if not self.settings.gifs_enabled:
                return "GIFs are off on this bot."
            if any(t == "image/gif" for _, t in result.images):
                return "Refused: one GIF per reply."
            gif, message = await webtools.find_gif(str(args.get("query") or ""), self.settings.klipy_api_key)
            if gif is not None:
                result.images.append((gif, "image/gif"))
            return message

        if name == "make_meme":
            if not self.settings.memes_enabled:
                return "Memes are off on this bot."
            if result.memes:
                return "Refused: one meme per reply."
            template = str(args.get("template") or "").strip()
            request_text = req.thread[-1].text if req and req.thread else ""
            subject = str(args.get("subject") or "").strip() or request_text
            # the bot's own messages, text and images alike (an image is stored as KIND_IMAGE,
            # not KIND_BOT, so the author is what tells them apart)
            own_text = "\n".join(m.text for m in (*req.window, *req.thread[:-1])
                                 if m.text and (m.is_bot or m.author_id == self.settings.phone_number)) if req else ""
            # once only: a meme about a person draws on what was said about them, which the bot's
            # replies also mention; refused every time, it spent all its tool rounds (2026-11-06)
            if result.recycle_refusals:
                own_text = ""
            png, message = await webtools.make_meme(template, args.get("lines"), subject=subject,
                                                    request_text=request_text, own_text=own_text)
            if png is None and message.startswith("These words come from your own"):
                result.recycle_refusals += 1
            if png is not None:
                result.images.append((png, "image/png"))
                result.memes += 1
                result.meme_subject = subject
            return message

        if name == "search_messages":
            query = str(args.get("query") or "").strip()
            if not query:
                return "A query is required."
            limit = max(1, min(int(args.get("limit") or 15), 50))
            after = None
            if args.get("since_hours"):
                after = int((time.time() - float(args["since_hours"]) * 3600) * 1000)
            if req.context_start is not None:  # a /clear forbids reaching past it
                after = max(after or 0, req.context_start)
            found = self.store.search(req.chat_id, query, limit=limit, after_ts=after, before_ts=req.thread[-1].ts)
            if not found:
                return "No message matches those words."
            return "\n".join(self._line(m, images) for m in found)

        if name == "remind":
            message = str(args.get("message") or "").strip()
            if not message:
                return "A message is required: what to post when the time comes."
            if len(message) > reminders.MAX_MESSAGE:
                return f"Message too long ({reminders.MAX_MESSAGE} characters at most)."
            due, problem = reminders.parse_when(args.get("at"), args.get("in_minutes"), self.tz, datetime.now(self.tz))
            if due is None:
                return problem
            if len(self.store.pending_reminders(req.chat_id)) >= reminders.MAX_PENDING:
                return f"Too many reminders waiting in this chat ({reminders.MAX_PENDING}): cancel one first."
            rid = self.store.add_reminder(req.chat_id, req.user_id, req.thread[-1].author_name, due, message,
                                          int(time.time() * 1000))
            return f"Reminder #{rid} set for {reminders.when_text(due, self.tz)}. It will be posted in this chat."

        if name == "list_reminders":
            if args.get("cancel_id") not in (None, ""):
                try:
                    cancel = int(args["cancel_id"])
                except (TypeError, ValueError):
                    return "cancel_id must be a reminder number."
                return (f"Reminder #{cancel} cancelled." if self.store.cancel_reminder(req.chat_id, cancel)
                        else f"There is no waiting reminder #{cancel} in this chat.")
            waiting = self.store.pending_reminders(req.chat_id)
            if not waiting:
                return "No reminder is waiting in this chat."
            return "\n".join(f"#{r['id']} {reminders.when_text(r['due_ts'], self.tz)} (asked by {r['author_name']}): {r['message']}"
                             for r in waiting)

        if name == "get_spend":
            now = datetime.now(self.tz)
            day = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
            month = int((now - timedelta(days=30)).timestamp() * 1000)
            return (f"Today: {self.store.cost_since(day, req.chat_id):.3f} $ here, "
                    f"{self.store.cost_since(day):.3f} $ in total. 30 days: "
                    f"{self.store.cost_since(month, req.chat_id):.3f} $ here, {self.store.cost_since(month):.3f} $ in total.")

        if self.settings.web_tools:
            out = await webtools.run(name, args, self.tz)
            if out is not None:
                return out

        return f"Unknown tool: {name}"

    # ── main loop ───────────────────────────────────────────────────────────

    @staticmethod
    def _fit(reply, budget: int, draft: str) -> str:
        """The final text within the budget: cut by max_tokens or still far too long, it ends after its last
        whole sentence. A shorter rewrite that came back empty (only a tool call, dropped) falls back on the
        long draft, cut."""
        text = reply.text
        if draft and not strip_tool_markup(text):
            text = draft
        words = word_count(strip_tool_markup(text))
        if reply.truncated or words > budget * CUT:
            cut = cut_to_words(text, int(budget * CUT), truncated=reply.truncated and text is reply.text)
            logger.info("answer cut from %d to %d words", words, word_count(cut))
            return cut
        return text

    async def run(self, req: AgentRequest, *, chat_model: str) -> AgentResult:
        images: dict[str, dict] = {}
        asked = req.thread[-1] if req.thread else None
        cls = length_class(asked.text if asked else "", with_image=bool(
            (asked and asked.images()) or (req.quoted and req.quoted.images())))
        messages = self._build_messages(req, images, budget=BUDGETS[cls])
        result = AgentResult(text="", chat_model=chat_model)
        tools = tools_for(self.settings)
        offered: list[dict] | None = tools
        recent = [m.text for m in (req.window + req.thread)[-8:] if not self._own(m) and m.text.strip()]
        names = tuple({m.author_name for m in (*req.window, *req.thread) if not self._own(m)})
        retried = False  # one more try per reply, whatever the reason: a local call takes seconds
        used: set[str] = set()
        announced = cls  # the class whose word count the model was given
        draft = ""       # a too long answer sent back for a shorter version

        for round_no in range(MAX_ROUNDS):
            now_cls = class_after_tools(cls, used)
            # no retry on the last round: the loop would end on it and its answer go out unchecked
            can_retry = not retried and round_no < MAX_ROUNDS - 1
            reply = await self.llm.chat(model=chat_model, messages=messages, temperature=0.7,
                                        max_tokens=min(self.settings.max_output_tokens, TOKEN_CAPS[now_cls]),
                                        tools=offered)
            result.chat_cost += reply.cost
            result.chat_model = reply.model
            if offered is not None and not reply.tool_calls and "<tool_call>" in (reply.text or ""):
                # written as text, not recognised by the server: run it anyway. Not when no tools were
                # offered (a shorter rewrite): nothing may run twice, a paid image least of all
                reply.tool_calls = tool_calls_in_text(reply.text)
                if reply.tool_calls:
                    logger.warning("tool call written as text, parsed: %s",
                                   ", ".join(c["function"]["name"] for c in reply.tool_calls))
                    reply.message = {"role": "assistant", "content": strip_tool_markup(reply.text) or None,
                                     "tool_calls": reply.tool_calls}
            if not reply.tool_calls:
                budget = BUDGETS[now_cls]
                problem = answer_problem(reply.text, self.settings.bot_name, recent, names) if can_retry else None
                if problem:
                    # the bad output is not kept in the conversation: the model would copy it again
                    retried = True
                    logger.warning("answer rejected (%s), asking again: %s", problem, reply.text[:120])
                    messages.append({"role": "user", "content": RETRY_NOTE.format(why=_WHY[problem])})
                    continue
                words = word_count(strip_tool_markup(reply.text))
                if can_retry and words > budget * OVER:
                    # kept, to be rewritten shorter; no tools on that call, so nothing runs twice
                    retried = True
                    offered = None
                    draft = reply.text
                    logger.info("answer of %d words for a budget of %d (%s), asking for a shorter one",
                                words, budget, now_cls)
                    messages.append({"role": "assistant", "content": reply.text})
                    messages.append({"role": "user", "content": SHORTEN_NOTE.format(words=words, budget=budget)})
                    continue
                result.text = self._fit(reply, budget, draft)
                break
            messages.append(reply.message)
            for call in reply.tool_calls:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):  # a small model may send a list or a bare value
                    args = {}
                logger.info("tool %s %s", fn.get("name"), json.dumps(args, ensure_ascii=False)[:200])
                used.add(str(fn.get("name")))
                try:
                    output = await self._run_tool(str(fn.get("name")), args, req, images, result)
                except (ValueError, TypeError, KeyError) as e:
                    if fn.get("name") == "generate_image":  # paid: an error after the call must not invite a retry
                        raise
                    # int("all"), float("24h")…: the model gets the error and can call again
                    logger.warning("tool %s failed on its arguments: %s", fn.get("name"), e)
                    output = f"Invalid arguments ({type(e).__name__}: {e}). Check the parameter types and call again."

                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": output})
            # a summary or a lookup raised the allowed length: the model was told less, say so now
            raised = class_after_tools(cls, used)
            if raised != announced and messages[-1]["role"] == "tool":
                announced = raised
                messages[-1]["content"] += f"\n\n(Your reply: at most {BUDGETS[raised]} words.)"
        else:
            # ran out of rounds while still calling tools: ask for the final answer without tools.
            # Without being told, the model writes one more tool call as text, and the server,
            # given no tools, lets it through as the answer.
            messages.append({"role": "user", "content": "No tool calls are left. Answer now, in plain text, "
                             "with what you already have."})
            now_cls = class_after_tools(cls, used)
            reply = await self.llm.chat(model=chat_model, messages=messages, temperature=0.7,
                                        max_tokens=min(self.settings.max_output_tokens, TOKEN_CAPS[now_cls]))
            result.chat_cost += reply.cost
            result.text = self._fit(reply, BUDGETS[now_cls], draft)

        # the prompt forbids the em dash and the model still writes one now and then
        result.text = re.sub(r"^[ \t]*—[ \t]*", "- ", result.text, flags=re.M)
        result.text = re.sub(r"[ \t]*—[ \t]*", ", ", result.text)
        if "<tool_call>" in result.text or "<function=" in result.text:
            logger.warning("tool call markup in the final answer, removed: %s", result.text[:120])
            result.text = strip_tool_markup(result.text) or ("" if result.images else NO_FINAL_ANSWER)
        # the model sometimes copies the "[HH:MM] Name : " header of the transcript it was shown
        result.text = re.sub(rf"^\s*(\[\d{{1,2}}:\d{{2}}\]\s*)?{re.escape(self.settings.bot_name)}\s*:\s*", "",
                             result.text, flags=re.IGNORECASE)
        problem = answer_problem(result.text, self.settings.bot_name, recent, names)
        if problem:  # still wrong after the second try: silence rather than the transcript or the reasoning
            logger.warning("answer rejected again (%s), not posted: %s", problem, result.text[:120])
            result.text = ""
        # after a meme, a text that only recycles the conversation (no word of the subject) is the
        # running joke leaking in again: the meme goes out alone rather than with an off-subject line
        if result.memes and result.text.strip() and result.meme_subject:
            said = webtools.meme_words(result.text)
            about = webtools.meme_words(result.meme_subject)
            chat = webtools.meme_words("\n".join(m.text for m in req.window if m.text))
            if chat & said and not (about & said):
                logger.info("text off the meme subject, dropped: %s", result.text[:80])
                result.text = ""
        if not result.text.strip() and not result.images:
            logger.warning("the model returned an empty answer")
        if result.text.strip() == SILENCE or (not result.text.strip() and not result.images):
            result.silent = not result.images
            result.text = ""
        return result
