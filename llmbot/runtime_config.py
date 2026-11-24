"""Settings that can be changed from the admin page, and how they are stored.

The `.env` stays the starting point. What is saved from the page goes to the bot's own
database (setting `config_overrides`) and wins over the `.env` at the next start too, until
it is reset from the page. Each field says whether a change applies at once or needs a restart."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from zoneinfo import ZoneInfo, available_timezones

from llmbot.config import MESSAGE_DEFAULTS, MESSAGE_PLACEHOLDERS, Settings
from llmbot.store import Store

OVERRIDES_KEY = "config_overrides"
PHONE_RE = re.compile(r"^\+\d{6,15}$")
# A Signal id: what a message carries instead of a number when the person hides theirs.
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

# Every tool the page can switch off, as the model sees its name.
TOOL_SWITCHES = ["generate_image", "send_gif", "make_meme", "get_spend", "web_search", "read_page", "get_weather", "get_time",
                 "calculate", "pick_random", "search_messages", "remind"]


@dataclass(frozen=True)
class Field:
    key: str
    kind: str                 # bool | int | text | url | list | phone | phones | secret | timezone
    live: bool = True         # False = needs a restart of the bot
    minimum: int = 0
    maximum: int = 0
    allow_empty: bool = True


FIELDS: dict[str, Field] = {f.key: f for f in [
    # provider
    Field("local_base_url", "url"),
    Field("chat_model", "text", allow_empty=False),
    Field("image_model", "text"),
    Field("transcribe_model", "text"),
    Field("transcribe_base_url", "url"),
    Field("openrouter_api_key", "secret"),
    # tools
    Field("web_tools", "bool"),
    Field("disabled_tools", "list"),
    Field("klipy_api_key", "secret"),
    # account and identity
    Field("phone_number", "phone", live=False, allow_empty=False),
    Field("bot_name", "text", allow_empty=False),
    Field("bot_aliases", "list"),
    Field("owner_number", "phone"),
    Field("allowed_groups", "list", live=False),
    Field("allowed_numbers", "phones", live=False),
    Field("timezone", "timezone"),
    Field("announce_startup", "bool", live=False),
    # behaviour
    Field("rate_limit_per_hour", "int", minimum=0, maximum=1000),
    Field("images_per_day", "int", minimum=0, maximum=1000),
    Field("context_messages", "int", minimum=1, maximum=200),
    Field("max_output_tokens", "int", minimum=50, maximum=8000),
    Field("describe_images", "bool"),
    Field("attachment_retention_days", "int", live=False, minimum=1, maximum=3650),
    # fixed messages
    *[Field(k, "message", allow_empty=False) for k in MESSAGE_DEFAULTS],
]}
SECRETS = {k for k, f in FIELDS.items() if f.kind == "secret"}


class ConfigError(ValueError):
    """A value the page sent is not acceptable; the message is shown to the user, in English."""


def coerce(key: str, raw):
    f = FIELDS.get(key)
    if f is None:
        raise ConfigError(f"Unknown setting: {key}")
    k = f.kind
    if k == "bool":
        if isinstance(raw, bool):
            return raw
        raise ConfigError("A yes or no answer is expected.")
    if k == "int":
        try:
            n = int(raw)
        except (TypeError, ValueError):
            raise ConfigError("A whole number is expected.") from None
        if not f.minimum <= n <= f.maximum:
            raise ConfigError(f"The number must be between {f.minimum} and {f.maximum}.")
        return n
    if k in ("list", "phones"):
        items = raw if isinstance(raw, list) else re.split(r"[,\n]", str(raw or ""))
        out = [str(i).strip() for i in items if str(i).strip()]
        if k == "phones" and (bad := [i for i in out if not (PHONE_RE.match(i) or UUID_RE.match(i))]):
            raise ConfigError(f"Invalid number: {bad[0]} (international format, for example +33612345678, "
                              "or a Signal ID for someone who hides their number).")
        return out
    text = str(raw if raw is not None else "").strip()
    if not text:
        if not f.allow_empty:
            raise ConfigError("This field cannot be empty.")
        return ""
    if k == "message":
        names = MESSAGE_PLACEHOLDERS.get(key, ())
        try:
            text.format(**{n: "x" for n in names})
        except (KeyError, IndexError, ValueError):
            usable = ", ".join("{" + n + "}" for n in names) or "none"
            raise ConfigError(f"Only these placeholders can be used in this message: {usable}. "
                              "Write a literal brace twice ({{ or }}).") from None
        return text
    if k == "url":
        if not re.match(r"^https?://[^\s/]+", text):
            raise ConfigError("An address starting with http:// or https:// is expected.")
        return text.rstrip("/")
    if k == "phone" and not PHONE_RE.match(text):
        raise ConfigError("Invalid number: international format expected, for example +33612345678.")
    if k == "timezone" and text not in available_timezones():
        raise ConfigError("Unknown timezone: use a name like Europe/Paris.")
    if k == "secret" and re.search(r"\s", text):
        raise ConfigError("A key contains no space or line break.")
    return text


def validate_disabled_tools(names: list[str]) -> list[str]:
    unknown = [n for n in names if n not in TOOL_SWITCHES]
    if unknown:
        raise ConfigError(f"Unknown tool: {unknown[0]}")
    return sorted(set(names), key=TOOL_SWITCHES.index)


def load(store: Store) -> dict:
    try:
        data = json.loads(store.get_setting(OVERRIDES_KEY) or "{}")
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in data.items() if k in FIELDS} if isinstance(data, dict) else {}


def save(store: Store, overrides: dict) -> None:
    store.set_setting(OVERRIDES_KEY, json.dumps(overrides, ensure_ascii=False) if overrides else None)


def apply(base: Settings, overrides: dict) -> Settings:
    """The settings the bot runs with: `base` (from the .env) with the saved overrides on top."""
    changes = {}
    for key, value in overrides.items():
        if key in FIELDS:
            changes[key] = list(value) if isinstance(value, list) else value
    return Settings(**{**base.__dict__, **changes})


def check(effective: Settings) -> list[str]:
    """Errors that forbid saving (the bot would have nothing to talk with) and returns warnings.

    Raises ConfigError for the former."""
    if not effective.local_base_url and not effective.openrouter_api_key:
        raise ConfigError("At least one provider is needed: a local server address or an OpenRouter key.")
    warnings = []
    if not effective.openrouter_api_key:
        if effective.image_model:
            warnings.append("Without an OpenRouter key, images cannot be created.")
        if effective.transcribe_model and not effective.transcribe_base_url:
            warnings.append("Without an OpenRouter key, voice messages will not be transcribed.")
    ZoneInfo(effective.timezone)
    return warnings


def changed_keys(before: Settings, after: Settings) -> list[str]:
    return [k for k in FIELDS if getattr(before, k) != getattr(after, k)]
