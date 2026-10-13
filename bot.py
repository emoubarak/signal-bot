import asyncio
import logging
import threading
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from signalbot import SignalBot

from llmbot.config import Settings
from llmbot.handler import LLMHandler
from llmbot.media import cleanup_attachments, ffmpeg_available
from llmbot.openrouter import BASE_URL, OpenRouter
from llmbot.personas import Registry
from llmbot.store import Store

ROOT = Path(__file__).parent


def heartbeat(push_url: str, signal_service: str) -> None:
    """Push "up" to Uptime Kuma every minute while signal-cli answers, "down" when it does not."""
    log = logging.getLogger("llmbot.heartbeat")
    while True:
        time.sleep(60)  # first push after signal-cli has had time to start
        try:
            ok = httpx.get(f"http://{signal_service}/v1/health", timeout=10).is_success
        except httpx.HTTPError:
            ok = False
        try:
            httpx.get(push_url, params={"status": "up" if ok else "down", "msg": "OK" if ok else "signal-cli unreachable"}, timeout=10)
        except httpx.HTTPError as exc:
            log.warning("kuma push failed: %s", exc)


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
    if not ffmpeg_available() and settings.transcription_enabled:
        log.warning("ffmpeg not found: voice notes are sent raw to the model (may fail on some formats)")

    llm = OpenRouter(
        settings.openrouter_api_key,
        base_url=settings.local_base_url or BASE_URL,
        local=settings.local_chat,
    )
    registry = Registry.load(ROOT / "prompts_custom.py")
    source = settings.chat_endpoint
    log.info("chat=%s (%s) image=%s transcribe=%s", settings.chat_model, source,
             settings.image_model or "off", settings.transcribe_model or "off")
    if settings.local_chat:
        log.info("local chat endpoint: %s", settings.local_base_url)
    if not settings.images_enabled:
        log.info("image generation is off (IMAGE_MODEL empty)")
    if not settings.transcription_enabled:
        log.info("voice note transcription is off (TRANSCRIBE_MODEL empty)")
    if settings.describe_images and not settings.openrouter_api_key:
        log.warning("DESCRIBE_IMAGES needs OpenRouter (OPENROUTER_API_KEY unset): image descriptions will fail")

    bot = SignalBot({
        "signal_service": settings.signal_service,
        "phone_number": settings.phone_number,
        "storage": {"type": "in-memory"},
        "logging_level": logging.INFO,
    })
    logging.getLogger("signalbot").propagate = False  # signalbot installs its own handler
    handler = LLMHandler(settings, store, llm, registry)
    bot.register(
        handler,
        contacts=settings.allowed_numbers or True,
        groups=settings.allowed_groups or True,
    )
    # On SIGTERM (a deploy), finish the answers in progress before signalbot cancels every task
    stop = bot.request_stop

    def request_stop() -> None:
        async def drain_then_stop() -> None:
            if handler.busy:
                log.info("stopping: waiting for %d answer(s) in progress", handler.busy)
                try:
                    await asyncio.wait_for(handler.drain(), 60)
                except TimeoutError:
                    log.warning("stopping: answers still in progress after 60 s, dropped")
            stop()
        bot.shutdown_drain = asyncio.get_running_loop().create_task(drain_then_stop())

    bot.request_stop = request_stop
    if settings.kuma_push_url:
        threading.Thread(target=heartbeat, args=(settings.kuma_push_url, settings.signal_service), daemon=True).start()
    bot.start()


if __name__ == "__main__":
    main()
