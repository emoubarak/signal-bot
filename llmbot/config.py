from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _list(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    signal_service: str
    phone_number: str
    openrouter_api_key: str

    chat_model: str = "anthropic/claude-haiku-5.5"
    image_model: str = "bytedance-seed/seedream-5-0-flash"
    transcribe_model: str = "google/gemini-3.1-flash-lite"

    bot_name: str = "Bot"
    # Other names people use to call the bot
    bot_aliases: list[str] = field(default_factory=list)
    data_dir: Path = Path("data")
    timezone: str = "Europe/Paris"

    # Empty list = everyone / every group. Groups: names or ids.
    allowed_groups: list[str] = field(default_factory=list)
    allowed_numbers: list[str] = field(default_factory=list)
    # Who may change the model at runtime. Empty = anyone.
    admin_numbers: list[str] = field(default_factory=list)

    rate_limit_per_hour: int = 20
    images_per_day: int = 10

    # Recent messages given to the model with every request (it can read more with its history tool)
    context_messages: int = 30
    max_output_tokens: int = 1000

    # Ask the chat model for a one-line description of every incoming image
    # so summaries know what was shared. Costs a vision call per image.
    describe_images: bool = False
    attachment_retention_days: int = 30

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        missing = [k for k in ("SIGNAL_SERVICE", "PHONE_NUMBER", "OPENROUTER_API_KEY") if not e.get(k)]
        if missing:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)} (see .env.example)")
        return cls(
            signal_service=e["SIGNAL_SERVICE"],
            phone_number=e["PHONE_NUMBER"],
            openrouter_api_key=e["OPENROUTER_API_KEY"],
            chat_model=e.get("OPENROUTER_MODEL") or cls.chat_model,
            image_model=e.get("IMAGE_MODEL") or cls.image_model,
            transcribe_model=e.get("TRANSCRIBE_MODEL") or cls.transcribe_model,
            bot_name=e.get("BOT_NAME") or cls.bot_name,
            bot_aliases=_list(e.get("BOT_ALIASES")),
            data_dir=Path(e.get("DATA_DIR") or "data"),
            timezone=e.get("TIMEZONE") or cls.timezone,
            allowed_groups=_list(e.get("ALLOWED_GROUPS")),
            allowed_numbers=_list(e.get("ALLOWED_NUMBERS")),
            admin_numbers=_list(e.get("ADMIN_NUMBERS")),
            rate_limit_per_hour=int(e.get("RATE_LIMIT_PER_HOUR") or cls.rate_limit_per_hour),
            images_per_day=int(e.get("IMAGES_PER_DAY") or cls.images_per_day),
            context_messages=int(e.get("CONTEXT_MESSAGES") or cls.context_messages),
            max_output_tokens=int(e.get("MAX_OUTPUT_TOKENS") or cls.max_output_tokens),
            describe_images=_bool(e.get("DESCRIBE_IMAGES"), cls.describe_images),
            attachment_retention_days=int(e.get("ATTACHMENT_RETENTION_DAYS") or cls.attachment_retention_days),
        )
