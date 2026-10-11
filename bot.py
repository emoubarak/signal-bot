import logging
from pathlib import Path

from dotenv import load_dotenv
from signalbot import SignalBot

from llmbot.config import Settings
from llmbot.handler import LLMHandler
from llmbot.media import cleanup_attachments, ffmpeg_available
from llmbot.openrouter import OpenRouter
from llmbot.personas import Registry
from llmbot.store import Store

ROOT = Path(__file__).parent


def main() -> None:
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log = logging.getLogger("llmbot")

    settings = Settings.from_env()
    data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
    settings = Settings(**{**settings.__dict__, "data_dir": data_dir})

    store = Store(data_dir / "bot.sqlite3")
    store.prune()
    removed = cleanup_attachments(data_dir, settings.attachment_retention_days)
    if removed:
        log.info("removed %d old attachments", removed)
    if not ffmpeg_available():
        log.warning("ffmpeg not found: voice notes are sent raw to the model (may fail on some formats)")

    llm = OpenRouter(settings.openrouter_api_key)
    registry = Registry.load(ROOT / "prompts_custom.py")
    log.info("chat=%s image=%s transcribe=%s", settings.chat_model, settings.image_model, settings.transcribe_model)

    bot = SignalBot({
        "signal_service": settings.signal_service,
        "phone_number": settings.phone_number,
        "storage": {"type": "in-memory"},
        "logging_level": logging.INFO,
    })
    logging.getLogger("signalbot").propagate = False  # signalbot installs its own handler
    bot.register(
        LLMHandler(settings, store, llm, registry),
        contacts=settings.allowed_numbers or True,
        groups=settings.allowed_groups or True,
    )
    bot.start()


if __name__ == "__main__":
    main()
