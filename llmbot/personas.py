"""Custom personas/styles from `prompts_custom.py`: every UPPERCASE string is
one the model can be asked to adopt by name (see prompts_custom.example.py)."""
from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("llmbot.personas")


@dataclass(frozen=True)
class Persona:
    name: str
    prompt: str


def load_custom(path: Path) -> list[Persona]:
    if not path.exists():
        return []
    try:
        spec = importlib.util.spec_from_file_location("prompts_custom", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        logger.exception("cannot load %s", path)
        return []
    found = [
        Persona(name=name.lower().replace("_", " "), prompt=value.strip())
        for name in dir(module)
        if name.isupper() and not name.startswith("_")
        and isinstance(value := getattr(module, name), str) and len(value.strip()) > 20
    ]
    logger.info("loaded %d personas from %s", len(found), path)
    return found


class Registry:
    def __init__(self, personas: list[Persona]) -> None:
        self.personas = personas

    @classmethod
    def load(cls, custom_path: Path | None = None) -> Registry:
        return cls(load_custom(custom_path) if custom_path else [])
