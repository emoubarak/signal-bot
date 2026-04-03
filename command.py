from typing import Optional, List, Tuple
from signalbot import Command, Context, regex_triggered
from llm.openai_client import run_chat
from utils.history import HistoryStore, extract_quote_info
import logging
import os
import importlib.util

from prompts import (
    ANALYSE,
    RESUME,
    ROAST,
    QUESTION,
    DEBAT,
    CONTEXT_INSTRUCTIONS,
    PERSO,
)

logger = logging.getLogger("commands.analyse")


def _load_custom_prompts() -> dict:
    """
    Load custom prompts from prompts_custom.py if it exists.
    Returns a dict of {PROMPT_NAME: prompt_string}.
    """
    custom_path = os.path.join(os.path.dirname(__file__), "prompts_custom.py")
    if not os.path.exists(custom_path):
        return {}

    try:
        spec = importlib.util.spec_from_file_location("prompts_custom", custom_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get all uppercase variables (prompts)
        custom_prompts = {}
        for name in dir(module):
            if name.isupper() and not name.startswith("_"):
                value = getattr(module, name)
                if isinstance(value, str) and len(value) > 20:  # Filter actual prompts
                    custom_prompts[name] = value

        logger.info(f"Loaded {len(custom_prompts)} custom prompts from prompts_custom.py")
        return custom_prompts
    except Exception as e:
        logger.warning(f"Failed to load custom prompts: {e}")
        return {}


# Load custom prompts at module level
CUSTOM_PROMPTS = _load_custom_prompts()


# ─── Command definitions ─────────────────────────────────────────────────────

# Base prompts (always available)
PROMPT_MAP = {
    "/analyse": (ANALYSE, "logical and structured analysis"),
    "/resume":  (RESUME, "summarize messages"),
    "/roast":   (ROAST, "roast the message author"),
    "/question": (QUESTION, "answer your question (neutral and factual)"),
    "/debat":   (DEBAT, "generate two opposing viewpoints"),
    "/perso":   (PERSO, "custom personality — /perso <description> : <message>"),
}

# Add custom prompts to PROMPT_MAP
for name, prompt in CUSTOM_PROMPTS.items():
    # Convert PROMPT_NAME to /prompt-name (underscores to hyphens, lowercase)
    cmd = "/" + name.lower().replace("_", "-")
    if cmd not in PROMPT_MAP:  # Don't override existing commands
        PROMPT_MAP[cmd] = (prompt, f"custom prompt: {name.lower().replace('_', ' ')}")

# Commands that work as "direct question" (history + question asked)
DIRECT_QUESTION_CMDS = {"/question"}

# Regex for all commands
_all_cmds = "|".join(k.lstrip("/") for k in PROMPT_MAP) + "|help"
CMD_REGEX = rf"^/({_all_cmds})(?:\s+.*)?$"


def _build_system_prompt(base_prompt: str) -> str:
    """Injects the context meta-prompt into the system prompt."""
    return base_prompt.strip() + "\n" + CONTEXT_INSTRUCTIONS.strip()


def _build_structured_payload(
    *,
    target_text: Optional[str] = None,
    target_author: Optional[str] = None,
    user_comment: Optional[str] = None,
    history_msgs: Optional[List[Tuple[str, str]]] = None,
    question_text: Optional[str] = None,
) -> str:
    """
    Builds a structured payload with clear markers for the LLM.
    Priority: TARGET MESSAGE > COMMENT > HISTORY.
    """
    parts = []

    if history_msgs:
        hist_block = HistoryStore.format_history_lines(history_msgs, sep="\n")
        parts.append(f"[HISTORY]\n{hist_block}")

    if target_text:
        if target_author:
            parts.append(f"[TARGET MESSAGE]\n{target_author}: {target_text}")
        else:
            parts.append(f"[TARGET MESSAGE]\n{target_text}")

    if user_comment:
        parts.append(f"[USER COMMENT]\n{user_comment}")

    if question_text:
        parts.append(f"[QUESTION]\n{question_text}")

    return "\n\n".join(parts) if parts else ""


class AnalyseCommand(Command):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def describe(self) -> str:
        return "Commands for analysis, summary, roast, Q&A, debate, and custom prompts."

    @regex_triggered(CMD_REGEX)
    async def handle(self, c: Context):
        cmd = (c.message.text or "").strip().split(maxsplit=1)[0].lower()
        logger.info(f"[{cmd}] Command received from {c.message.source}")

        if cmd == "/help":
            await self._handle_help(c)
            return

        prompt_entry = PROMPT_MAP.get(cmd)
        if not prompt_entry:
            logger.warning(f"[{cmd}] Unknown command")
            await c.reply("Unknown command.")
            return

        if cmd == "/perso":
            await self._handle_perso(c)
            return

        max_ctx = self.config["limits"]["max_context_messages"]
        hard_limit = self.config["limits"]["hard_input_char_limit"]

        parts = (c.message.text or "").strip().split(maxsplit=1)
        arg: Optional[str] = parts[1] if len(parts) > 1 else None

        quote_text, quote_author = extract_quote_info(c)

        user_payload: Optional[str] = None

        # Direct question commands
        if cmd in DIRECT_QUESTION_CMDS:
            user_payload = await self._build_direct_payload(
                c, cmd, arg, quote_text, quote_author, max_ctx, hard_limit
            )
            if user_payload is None:
                return
        # Debate
        elif cmd == "/debat":
            user_payload = await self._build_analysis_payload(
                c, cmd, arg, quote_text, quote_author, max_ctx
            )
            if user_payload is None:
                return
        # Other commands (analyse, resume, roast, custom prompts)
        else:
            user_payload = await self._build_analysis_payload(
                c, cmd, arg, quote_text, quote_author, max_ctx
            )
            if user_payload is None:
                return

        if not user_payload:
            await c.reply("Reply to a message or use `/command N` to target the last N messages.")
            return

        if len(user_payload) > hard_limit:
            await c.reply("Context too long. Reduce the number of messages.")
            return

        logger.info(f"[{cmd}] Sending to LLM.")
        await c.start_typing()
        try:
            logger.info(f"[{cmd}] Payload: {user_payload[:300]}{'...' if len(user_payload) > 300 else ''}")
            prompt_str, _ = PROMPT_MAP.get(cmd)
            system = _build_system_prompt(prompt_str)

            answer = await run_chat(
                system_prompt=system,
                user_content=user_payload,
                model=self.config["openai"]["model"],
                temperature=self.config["openai"]["temperature"],
                max_tokens=self.config["openai"]["max_output_tokens"],
            )
        finally:
            await c.stop_typing()

        await self._send_and_record(c, answer)

    # ─── Sub-handlers ──────────────────────────────────────────────────────────

    async def _handle_help(self, c: Context):
        """Displays categorized help."""
        max_ctx = self.config["limits"]["max_context_messages"]

        lines = ["\n📋 Available commands:\n"]
        lines.append("🔍 Analysis & Utilities")
        for cmd in ["/analyse", "/resume", "/roast", "/question", "/debat"]:
            if cmd in PROMPT_MAP:
                _, desc = PROMPT_MAP[cmd]
                lines.append(f"  {cmd} — {desc}")

        lines.append("\n🎭 Personality")
        lines.append("  /perso — custom personality — /perso <description> : <message>")

        # Show custom prompts if any
        custom_cmds = [k for k in PROMPT_MAP if k not in ["/analyse", "/resume", "/roast", "/question", "/debat", "/perso"]]
        if custom_cmds:
            lines.append("\n🎨 Custom prompts")
            for cmd in sorted(custom_cmds):
                _, desc = PROMPT_MAP[cmd]
                lines.append(f"  {cmd} — {desc}")

        mechanics = (
            "\n\n🛠️ How it works:\n"
            "• Reply to a message → targets THAT message\n"
            "• /command N → targets the last N messages\n"
            "• /command (no argument) → uses full history\n"
            f"• Max history: {max_ctx} messages\n"
            "• /question [N] <your question> → ask a question with context\n"
            "• /perso <description> : <message> → custom personality"
        )

        await c.reply("\n".join(lines) + mechanics)

    async def _handle_perso(self, c: Context):
        """Handles /perso <description> : <message or quote>."""
        parts = (c.message.text or "").strip().split(maxsplit=1)
        arg = parts[1] if len(parts) > 1 else None

        if not arg or ":" not in arg:
            await c.reply(
                "Usage: /perso <character description> : <message or question>\n"
                "Example: /perso a drunk pirate: what's the meaning of life?"
            )
            return

        persona_desc, user_msg = arg.split(":", 1)
        persona_desc = persona_desc.strip()
        user_msg = user_msg.strip()

        quote_text, quote_author = extract_quote_info(c)
        if not user_msg and quote_text:
            user_msg = quote_text

        if not user_msg:
            await c.reply("I need a message after ':' or reply to a message.")
            return

        system = _build_system_prompt(
            f"You adopt the following personality: {persona_desc}\n\n{PERSO}"
        )

        payload = _build_structured_payload(target_text=user_msg)

        hard_limit = self.config["limits"]["hard_input_char_limit"]
        if len(payload) > hard_limit:
            await c.reply("Message too long.")
            return

        await c.start_typing()
        try:
            answer = await run_chat(
                system_prompt=system,
                user_content=payload,
                model=self.config["openai"]["model"],
                temperature=self.config["openai"]["temperature"],
                max_tokens=self.config["openai"]["max_output_tokens"],
            )
        finally:
            await c.stop_typing()

        await self._send_and_record(c, answer)

    async def _build_direct_payload(
        self, c: Context, cmd: str, arg: Optional[str],
        quote_text: Optional[str], quote_author: Optional[str],
        max_ctx: int, hard_limit: int,
    ) -> Optional[str]:
        """
        Builds payload for "direct" commands (/question).
        These treat the argument as a question/message from the user,
        with history as secondary context.
        """
        if not arg or not arg.strip():
            if quote_text:
                return _build_structured_payload(
                    target_text=quote_text,
                    target_author=quote_author,
                )

            if cmd == "/question":
                await c.reply(
                    "Usage: /question [N] <your question>.\n"
                    "Examples:\n"
                    "- /question Does walking burn calories?\n"
                    "- /question 12 Should I reduce carbs?"
                )
            else:
                await c.reply(f"Usage: {cmd} <your message>")
            return None

        tokens = arg.strip().split(maxsplit=1)
        first = tokens[0]

        if first.isdigit():
            n = min(int(first), max_ctx)
            question_text = tokens[1].strip() if len(tokens) > 1 else ""
        else:
            n = max_ctx
            question_text = arg.strip()

        if not question_text:
            await c.reply(f"Your message is empty. Example: {cmd} 8 What does NXDOMAIN mean?")
            return None

        msgs = HistoryStore.get_recent_messages(c, n=n, exclude_current=True)

        if quote_text:
            payload = _build_structured_payload(
                target_text=quote_text,
                target_author=quote_author,
                history_msgs=msgs if msgs else None,
                question_text=question_text,
            )
        else:
            payload = _build_structured_payload(
                history_msgs=msgs if msgs else None,
                question_text=question_text,
            )

        if len(payload) > hard_limit and msgs:
            kept: List[Tuple[str, str]] = list(msgs)
            while kept:
                if quote_text:
                    candidate = _build_structured_payload(
                        target_text=quote_text,
                        target_author=quote_author,
                        history_msgs=kept,
                        question_text=question_text,
                    )
                else:
                    candidate = _build_structured_payload(
                        history_msgs=kept,
                        question_text=question_text,
                    )
                if len(candidate) <= hard_limit:
                    payload = candidate
                    break
                kept = kept[1:]
            if not kept:
                payload = _build_structured_payload(question_text=question_text)

        return payload

    async def _build_analysis_payload(
        self, c: Context, cmd: str, arg: Optional[str],
        quote_text: Optional[str], quote_author: Optional[str],
        max_ctx: int,
    ) -> Optional[str]:
        """
        Builds payload for analysis commands (/analyse, /resume, /roast, etc.).
        Priority: quote > N messages > full history.
        Most recent message is marked as target.
        """
        if quote_text:
            user_comment = arg if arg and not arg.isdigit() else None
            return _build_structured_payload(
                target_text=quote_text,
                target_author=quote_author,
                user_comment=user_comment,
            )

        if arg and arg.isdigit():
            n = min(int(arg), max_ctx)
            msgs = HistoryStore.get_recent_messages(c, n=n, exclude_current=True)
            if not msgs:
                await c.reply("No usable history here.")
                return None

            if len(msgs) > 1:
                history_msgs = msgs[:-1]
                target_name, target_text = msgs[-1]
                return _build_structured_payload(
                    target_text=target_text,
                    target_author=target_name,
                    history_msgs=history_msgs,
                )
            else:
                target_name, target_text = msgs[0]
                return _build_structured_payload(
                    target_text=target_text,
                    target_author=target_name,
                )

        else:
            msgs = HistoryStore.get_recent_messages(c, n=max_ctx, exclude_current=True)
            if not msgs:
                await c.reply("No previous message found.")
                return None

            if len(msgs) > 1:
                history_msgs = msgs[:-1]
                target_name, target_text = msgs[-1]
                return _build_structured_payload(
                    target_text=target_text,
                    target_author=target_name,
                    history_msgs=history_msgs,
                )
            else:
                target_name, target_text = msgs[0]
                return _build_structured_payload(
                    target_text=target_text,
                    target_author=target_name,
                )

    # ─── Send + record ─────────────────────────────────────────────────────────

    async def _send_and_record(self, c: Context, answer: str):
        """Sends the reply and records it in history."""
        bot_name = (
            self.config.get("bot_name")
            or (getattr(getattr(c, "bot", None), "name", None))
            or "Bot"
        )
        if getattr(c.message, "group", None):
            await c.reply(answer)
            HistoryStore.add(c, answer, name_override=bot_name)
        else:
            await c.bot.send(c.message.source, answer)
            HistoryStore.add(c, answer, name_override=bot_name)