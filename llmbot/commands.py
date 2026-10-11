"""Prompt-based commands (built-in and custom) and argument parsing."""
from __future__ import annotations

import importlib.util
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from llmbot import prompts

logger = logging.getLogger("llmbot.commands")


@dataclass(frozen=True)
class PromptCommand:
    name: str
    description: str
    prompt: str
    temperature: float = 0.7
    # The last message of someone else becomes the [CIBLE] when nothing is quoted.
    needs_target: bool = False
    # Default window = everything since the last call of this command in the chat.
    since_last: bool = False
    default_window: int = 15
    custom: bool = False


BUILTIN: list[PromptCommand] = [
    PromptCommand("resume", "résume la conversation depuis le dernier /resume, ou les N derniers messages",
                  prompts.RESUME, temperature=0.3, since_last=True, default_window=50),
    PromptCommand("analyse", "analyse un message (en réponse) ou l'échange récent",
                  prompts.ANALYSE, temperature=0.4, needs_target=True),
    PromptCommand("roast", "roaste l'auteur du message (en réponse, sinon le dernier message)",
                  prompts.ROAST, temperature=1.0, needs_target=True),
    PromptCommand("debat", "deux points de vue opposés sur un message",
                  prompts.DEBAT, temperature=0.9, needs_target=True),
    PromptCommand("question", "question neutre et factuelle, la conversation sert de contexte",
                  prompts.QUESTION, temperature=0.2, default_window=20),
]


def load_custom(path: Path) -> list[PromptCommand]:
    """Every UPPERCASE string in `prompts_custom.py` becomes /lower-case-name."""
    if not path.exists():
        return []
    try:
        spec = importlib.util.spec_from_file_location("prompts_custom", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        logger.exception("cannot load %s", path)
        return []
    found: list[PromptCommand] = []
    for name in dir(module):
        value = getattr(module, name)
        if name.isupper() and not name.startswith("_") and isinstance(value, str) and len(value.strip()) > 20:
            first_line = value.strip().splitlines()[0].strip()
            found.append(PromptCommand(
                name=name.lower().replace("_", "-"),
                description=f"prompt personnalisé : {first_line[:60]}",
                prompt=value.strip(),
                temperature=0.8,
                needs_target=True,
                default_window=10,
                custom=True,
            ))
    logger.info("loaded %d custom prompts from %s", len(found), path)
    return found


class Registry:
    def __init__(self, commands: list[PromptCommand]) -> None:
        self._by_name = {c.name: c for c in commands}

    @classmethod
    def load(cls, custom_path: Path | None = None) -> Registry:
        commands = list(BUILTIN)
        taken = {c.name for c in commands}
        for custom in load_custom(custom_path) if custom_path else []:
            if custom.name in taken:
                logger.warning("custom prompt %s ignored: name already used", custom.name)
                continue
            commands.append(custom)
            taken.add(custom.name)
        return cls(commands)

    def get(self, name: str) -> PromptCommand | None:
        return self._by_name.get(name)

    def builtin(self) -> list[PromptCommand]:
        return [c for c in self._by_name.values() if not c.custom]

    def custom(self) -> list[PromptCommand]:
        return [c for c in self._by_name.values() if c.custom]


# ── argument parsing ────────────────────────────────────────────────────────

_CMD_RE = re.compile(r"^/(?P<name>[\w-]+)(?:@\S+)?(?:\s+(?P<rest>.*))?$", re.S)


def split_command(text: str) -> tuple[str, str] | None:
    """'/Roast 5 trop long' -> ('roast', '5 trop long'). None if not a command."""
    m = _CMD_RE.match(text.strip())
    if not m:
        return None
    return m.group("name").lower(), (m.group("rest") or "").strip()


def parse_count_and_request(rest: str) -> tuple[int | None, str]:
    """'12 pourquoi ?' -> (12, 'pourquoi ?'); 'pourquoi ?' -> (None, 'pourquoi ?')."""
    parts = rest.split(maxsplit=1)
    if parts and parts[0].isdigit():
        return int(parts[0]), (parts[1].strip() if len(parts) > 1 else "")
    return None, rest.strip()


ASPECT_RATIOS = {"1:1", "1:2", "2:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "9:21", "auto"}


def parse_image_args(rest: str) -> tuple[str | None, str]:
    """'16:9 un chat' -> ('16:9', 'un chat'); 'un chat' -> (None, 'un chat')."""
    parts = rest.split(maxsplit=1)
    if parts and parts[0].lower() in ASPECT_RATIOS:
        return parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
    return None, rest.strip()
