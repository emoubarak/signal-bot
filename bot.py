import os
import logging

from signalbot import SignalBot
from command import AnalyseCommand
from utils.history import HistoryRecorder
from config import BOT_CONFIG
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)


def main() -> None:
    # Env variables are described in README.md and .env.example
    service = os.environ["SIGNAL_SERVICE"]  # e.g., "127.0.0.1:8080"
    phone = os.environ["PHONE_NUMBER"]      # e.g., "+33700000000"

    bot = SignalBot({
        "signal_service": service,
        "phone_number": phone,
        # tip: you can also pass "consumers" if doing heavy blocking I/O
        # "consumers": 5,
    })

    # History recorder (always first)
    bot.register(HistoryRecorder(max_messages_per_chat=300))

    # LLM Commands
    bot.register(AnalyseCommand(BOT_CONFIG))

    bot.start()


if __name__ == "__main__":
    main()