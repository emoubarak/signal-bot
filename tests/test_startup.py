"""The restart announcement: which model it names, and where it goes."""
from llmbot.config import Settings
from llmbot.openrouter import ChatResult, OpenRouterError
from llmbot.startup import StartupAnnouncer


def settings(**overrides) -> Settings:
    base = {"signal_service": "127.0.0.1:8080", "phone_number": "+33700000000",
            "bot_name": "Mop", "image_model": "", "transcribe_model": ""}
    return Settings(**{**base, **overrides})


class ProbeLLM:
    """Answers the startup probe with whatever model the server reports."""

    def __init__(self, served: str = "qwen3.8-flash-next-q2_0", fail: bool = False) -> None:
        self.served = served
        self.fail = fail
        self.calls = []

    async def chat(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "max_tokens": kwargs.get("max_tokens")})
        if self.fail:
            raise OpenRouterError("connection refused")
        return ChatResult(text="", finish_reason="stop", model=self.served)


class FakeBot:
    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.fail_for = fail_for or set()
        self.messages = self
        self.sent: list[tuple[str, str]] = []

    async def send(self, message, recipient):
        if recipient in self.fail_for:
            raise RuntimeError("unreachable")
        self.sent.append((recipient, message.text))
        return None


class FakeContext:
    def __init__(self, bot) -> None:
        self.bot = bot


async def test_announcement_names_the_model_actually_served():
    """A llama-swap selector answers as the model it loaded, not as its own name."""
    llm = ProbeLLM(served="qwen3.8-flash-next-q2_0")
    announcer = StartupAnnouncer(settings(local_base_url="http://x:8089/v1", chat_model="qwen3.6-35b-ablit",
                                            owner_number="+33628508830"), llm)
    bot = FakeBot()

    await announcer.handle_ready(FakeContext(bot))

    assert llm.calls[0]["model"] == "qwen3.6-35b-ablit"  # asked under its configured name
    assert llm.calls[0]["max_tokens"] == 1  # the probe is a cheap one-token call
    text = bot.sent[0][1]
    assert "qwen3.8-flash-next-q2_0" in text
    assert "en local" in text


async def test_announcement_says_openrouter_when_chat_is_not_local():
    announcer = StartupAnnouncer(settings(openrouter_api_key="k", chat_model="deepseek/deepseek-v4.1-flash",
                                          owner_number="+33628508830"),
                                 ProbeLLM(served="deepseek/deepseek-v4.1-flash"))
    bot = FakeBot()

    await announcer.handle_ready(FakeContext(bot))

    assert "sur OpenRouter" in bot.sent[0][1]
    assert "en local" not in bot.sent[0][1]


async def test_probe_failure_still_announces_the_configured_name():
    """A model that is slow to load must not silence the announcement."""
    announcer = StartupAnnouncer(settings(local_base_url="http://x:8089/v1", chat_model="qwen3.6-35b-ablit",
                                           owner_number="+33628508830"),
                                 ProbeLLM(fail=True))
    bot = FakeBot()

    await announcer.handle_ready(FakeContext(bot))

    assert "qwen3.6-35b-ablit" in bot.sent[0][1]


async def test_announcement_goes_to_every_allowed_group_and_the_owner():
    announcer = StartupAnnouncer(
        settings(allowed_groups=["LE SILENCIER", "Famille"], owner_number="+33628508830",
                 local_base_url="http://x:8089/v1"),
        ProbeLLM())
    bot = FakeBot()

    await announcer.handle_ready(FakeContext(bot))

    assert [r for r, _ in bot.sent] == ["LE SILENCIER", "Famille", "+33628508830"]


async def test_owner_only_when_no_group_is_configured():
    announcer = StartupAnnouncer(
        settings(owner_number="+33628508830", local_base_url="http://x:8089/v1"), ProbeLLM())
    bot = FakeBot()

    await announcer.handle_ready(FakeContext(bot))

    assert [r for r, _ in bot.sent] == ["+33628508830"]


async def test_unreachable_chat_does_not_stop_the_others():
    announcer = StartupAnnouncer(
        settings(allowed_groups=["Famille", "Morte"], owner_number="+33628508830",
                 local_base_url="http://x:8089/v1"),
        ProbeLLM())
    bot = FakeBot(fail_for={"Morte"})

    await announcer.handle_ready(FakeContext(bot))

    assert [r for r, _ in bot.sent] == ["Famille", "+33628508830"]


async def test_nothing_is_sent_without_a_destination():
    announcer = StartupAnnouncer(settings(local_base_url="http://x:8089/v1"), ProbeLLM())
    bot = FakeBot()

    await announcer.handle_ready(FakeContext(bot))

    assert bot.sent == []


def test_startup_announcement_can_be_switched_off():
    env = {"SIGNAL_SERVICE": "s", "PHONE_NUMBER": "p", "OPENROUTER_API_KEY": "k"}
    assert Settings.from_env(env).announce_startup is True
    assert Settings.from_env({**env, "ANNOUNCE_STARTUP": "false"}).announce_startup is False


def test_owner_number_is_read_and_trimmed():
    env = {"SIGNAL_SERVICE": "s", "PHONE_NUMBER": "p", "OPENROUTER_API_KEY": "k"}
    assert Settings.from_env({**env, "OWNER_NUMBER": " +33628508830 "}).owner_number == "+33628508830"
    assert Settings.from_env(env).owner_number == ""
