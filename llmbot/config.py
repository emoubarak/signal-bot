from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
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
    # OpenRouter is only needed for what a local server cannot do (images, voice notes)
    # and as the chat endpoint when LOCAL_BASE_URL is not set.
    openrouter_api_key: str = ""

    chat_model: str = "deepseek/deepseek-v4.1-flash"
    image_model: str = "meta/muse-image"
    transcribe_model: str = "google/gemini-3.1-flash-lite"

    # Local OpenAI-compatible chat endpoint (llama-swap, llama-server, vLLM…).
    # When set, chat goes there instead of OpenRouter, and it needs no key.
    local_base_url: str = ""

    bot_name: str = "Bot"
    # Other names people use to call the bot
    bot_aliases: list[str] = field(default_factory=list)
    data_dir: Path = Path("data")
    timezone: str = "Europe/Paris"

    # Empty list = everyone / every group. Groups: names or ids.
    allowed_groups: list[str] = field(default_factory=list)
    allowed_numbers: list[str] = field(default_factory=list)

    # The owner's number: told which model the bot restarted on, and the private
    # chat the startup message goes to. Empty = no private announcement.
    owner_number: str = ""

    rate_limit_per_hour: int = 20
    images_per_day: int = 10

    # Recent messages given to the model with every request (it can read more with its history tool)
    context_messages: int = 30
    max_output_tokens: int = 1000

    # Ask the chat model for a one-line description of every incoming image
    # so summaries know what was shared. Costs a vision call per image.
    describe_images: bool = False
    attachment_retention_days: int = 30

    # Free tools without any account: web search, reading a link, weather, time, calculator.
    web_tools: bool = True
    # Free key from partner.klipy.com. Empty = the send_gif tool is not offered.
    klipy_api_key: str = ""
    # Tools switched off one by one from the admin page (names as the model sees them).
    disabled_tools: list[str] = field(default_factory=list)

    # Admin web page: only started when a token is set. Env only, never editable from the page itself.
    admin_token: str = ""
    admin_host: str = "127.0.0.1"
    admin_port: int = 8090

    # Uptime Kuma push URL, pinged every minute while signal-cli answers. Empty = off.
    kuma_push_url: str = ""

    # Say "I restarted" in private to the owner.
    announce_startup: bool = True

    @property
    def local_chat(self) -> bool:
        return bool(self.local_base_url)

    @property
    def chat_endpoint(self) -> str:
        return self.local_base_url or "openrouter"

    @property
    def images_enabled(self) -> bool:
        return bool(self.image_model) and "generate_image" not in self.disabled_tools

    @property
    def gifs_enabled(self) -> bool:
        return bool(self.klipy_api_key) and "send_gif" not in self.disabled_tools

    def tool_enabled(self, name: str) -> bool:
        return name not in self.disabled_tools

    @property
    def transcription_enabled(self) -> bool:
        return bool(self.transcribe_model)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        local_base_url = (e.get("LOCAL_BASE_URL") or "").rstrip("/")
        api_key = e.get("OPENROUTER_API_KEY") or ""
        missing = [k for k in ("SIGNAL_SERVICE", "PHONE_NUMBER") if not e.get(k)]
        if not local_base_url and not api_key:
            missing.append("OPENROUTER_API_KEY (or LOCAL_BASE_URL for a local model)")
        if missing:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)} (see .env.example)")
        return cls(
            signal_service=e["SIGNAL_SERVICE"],
            phone_number=e["PHONE_NUMBER"],
            openrouter_api_key=api_key,
            chat_model=e.get("CHAT_MODEL") or e.get("OPENROUTER_MODEL") or cls.chat_model,
            image_model=e.get("IMAGE_MODEL", cls.image_model),
            transcribe_model=e.get("TRANSCRIBE_MODEL", cls.transcribe_model),
            local_base_url=local_base_url,
            bot_name=e.get("BOT_NAME") or cls.bot_name,
            bot_aliases=_list(e.get("BOT_ALIASES")),
            data_dir=Path(e.get("DATA_DIR") or "data"),
            timezone=e.get("TIMEZONE") or cls.timezone,
            allowed_groups=_list(e.get("ALLOWED_GROUPS")),
            allowed_numbers=_list(e.get("ALLOWED_NUMBERS")),
            owner_number=(e.get("OWNER_NUMBER") or "").strip(),
            rate_limit_per_hour=int(e.get("RATE_LIMIT_PER_HOUR") or cls.rate_limit_per_hour),
            images_per_day=int(e.get("IMAGES_PER_DAY") or cls.images_per_day),
            context_messages=int(e.get("CONTEXT_MESSAGES") or cls.context_messages),
            max_output_tokens=int(e.get("MAX_OUTPUT_TOKENS") or cls.max_output_tokens),
            describe_images=_bool(e.get("DESCRIBE_IMAGES"), cls.describe_images),
            web_tools=_bool(e.get("WEB_TOOLS"), cls.web_tools),
            klipy_api_key=(e.get("KLIPY_API_KEY") or "").strip(),
            disabled_tools=_list(e.get("DISABLED_TOOLS")),
            admin_token=(e.get("ADMIN_TOKEN") or "").strip(),
            admin_host=e.get("ADMIN_HOST") or cls.admin_host,
            admin_port=int(e.get("ADMIN_PORT") or cls.admin_port),
            attachment_retention_days=int(e.get("ATTACHMENT_RETENTION_DAYS") or cls.attachment_retention_days),
            kuma_push_url=e.get("KUMA_PUSH_URL") or "",
            announce_startup=_bool(e.get("ANNOUNCE_STARTUP"), cls.announce_startup),
        )


class LiveSettings:
    """The settings the running bot reads, which the admin page can change.

    Every `settings.x` read goes to the current snapshot, so the handler, the agent and the
    announcer (which all hold this object) see a change on their next request without being
    rebuilt. A snapshot itself stays immutable: a request never sees half an update."""

    def __init__(self, base: Settings) -> None:
        object.__setattr__(self, "_snapshot", base)

    @property
    def snapshot(self) -> Settings:
        return self._snapshot

    def replace_with(self, new: Settings) -> None:
        object.__setattr__(self, "_snapshot", new)

    def update(self, **changes) -> Settings:
        new = replace(self._snapshot, **changes)
        object.__setattr__(self, "_snapshot", new)
        return new

    def __getattr__(self, name: str):
        return getattr(self._snapshot, name)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("use update(): a snapshot is replaced as a whole")
