"""Searching old messages, reminders, and drawing at random."""
import asyncio
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from llmbot import reminders, webtools
from llmbot.agent import Agent, AgentRequest
from llmbot.config import Settings
from llmbot.store import KIND_TEXT, Store, StoredMessage

TZ = ZoneInfo("Europe/Paris")
NOW = datetime(2026, 10, 16, 19, 30, tzinfo=TZ)
NOW_MS = int(NOW.timestamp() * 1000)


def settings(**overrides) -> Settings:
    base = {"signal_service": "s", "phone_number": "+33700000000", "openrouter_api_key": "k"}
    return Settings(**{**base, **overrides})


def msg(ts, text, author="Karim", chat="g"):
    return StoredMessage(chat_id=chat, ts=ts, author_id=author, author_name=author, kind=KIND_TEXT, text=text)


def make_agent(store):
    return Agent(settings(), store=store, llm=None, registry=None, tz=TZ)


def request(store, chat="g", context_start=None):
    last = msg(10_000_000, "Sig help", chat=chat)
    store.add(last)
    return AgentRequest(chat_id=chat, chat_name="g", user_id="u1", thread=[last], window=[], quoted=None,
                        images_remaining=3, context_start=context_start)


# ── search_messages ─────────────────────────────────────────────────────────

async def test_search_finds_by_all_words_ignoring_case_and_accents():
    store = Store()
    store.add(msg(1000, "On mange au Restaurant Étoilé samedi ?"))
    store.add(msg(2000, "Le restaurant de Lyon était nul"))
    store.add(msg(3000, "Qui vient samedi ?"))
    agent, req = make_agent(store), request(store)
    out = await agent._run_tool("search_messages", {"query": "RESTAURANT etoile"}, req, {}, None)
    assert "Étoilé" in out and "de Lyon" not in out
    out = await agent._run_tool("search_messages", {"query": "restaurant"}, req, {}, None)
    assert out.count("\n") == 1   # two matches, in reading order
    assert out.index("Étoilé") < out.index("de Lyon")


async def test_search_stays_in_its_chat_and_stops_at_a_clear():
    store = Store()
    store.add(msg(1000, "secret plan", chat="other"))
    store.add(msg(2000, "plan before the clear"))
    store.add(msg(6000, "plan after the clear"))
    agent = make_agent(store)
    out = await agent._run_tool("search_messages", {"query": "plan"}, request(store, context_start=5000), {}, None)
    assert "after the clear" in out and "before" not in out and "secret" not in out


async def test_search_with_nothing_found_or_no_query():
    store = Store()
    agent, req = make_agent(store), request(store)
    assert await agent._run_tool("search_messages", {"query": "zzz"}, req, {}, None) == "No message matches those words."
    assert await agent._run_tool("search_messages", {"query": "  "}, req, {}, None) == "A query is required."


async def test_search_limit_is_capped():
    store = Store()
    for i in range(80):
        store.add(msg(1000 + i, f"pizza {i}"))
    out = await make_agent(store)._run_tool("search_messages", {"query": "pizza", "limit": 500}, request(store), {}, None)
    assert out.count("\n") == 49 and "pizza 79" in out   # 50 newest matches


# ── reminders: setting them ─────────────────────────────────────────────────

@pytest.fixture
def clock(monkeypatch):
    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz is None else NOW.astimezone(tz)
    monkeypatch.setattr("llmbot.agent.datetime", FakeDatetime)
    return NOW


async def test_remind_in_minutes(clock):
    store = Store()
    out = await make_agent(store)._run_tool("remind", {"message": "Call the plumber", "in_minutes": 20}, request(store), {}, None)
    assert out.startswith("Reminder #1 set for Friday 16/10/2026 19:50")
    [r] = store.pending_reminders("g")
    assert r["due_ts"] == NOW_MS + 20 * 60_000 and r["message"] == "Call the plumber" and r["author_name"] == "Karim"


async def test_remind_at_a_clock_time_in_the_groups_timezone(clock):
    store = Store()
    await make_agent(store)._run_tool("remind", {"message": "dinner", "at": "2026-10-17 20:00"}, request(store), {}, None)
    [r] = store.pending_reminders("g")
    assert datetime.fromtimestamp(r["due_ts"] / 1000, TZ).strftime("%Y-%m-%d %H:%M") == "2026-10-17 20:00"


@pytest.mark.parametrize("args,fragment", [
    ({"message": "x", "at": "2026-10-16 18:00"}, "already past"),
    ({"message": "x", "in_minutes": -5}, "already past"),
    ({"message": "x", "at": "next tuesday"}, "Could not read"),
    ({"message": "x", "in_minutes": "soon"}, "Could not read"),
    ({"message": "x"}, "Give the time"),
    ({"message": "x", "at": "2028-01-01 10:00"}, "Too far"),
    ({"message": " ", "in_minutes": 5}, "message is required"),
    ({"message": "y" * 500, "in_minutes": 5}, "too long"),
])
async def test_remind_refuses_what_makes_no_sense(clock, args, fragment):
    store = Store()
    out = await make_agent(store)._run_tool("remind", args, request(store), {}, None)
    assert fragment in out and store.pending_reminders("g") == []


async def test_remind_has_a_ceiling_per_chat(clock):
    store = Store()
    agent, req = make_agent(store), request(store)
    for i in range(reminders.MAX_PENDING):
        await agent._run_tool("remind", {"message": f"r{i}", "in_minutes": 10 + i}, req, {}, None)
    out = await agent._run_tool("remind", {"message": "one more", "in_minutes": 5}, req, {}, None)
    assert "Too many reminders" in out
    other = await agent._run_tool("remind", {"message": "elsewhere", "in_minutes": 5}, request(store, chat="h"), {}, None)
    assert other.startswith("Reminder #")


async def test_list_and_cancel_reminders(clock):
    store = Store()
    agent, req = make_agent(store), request(store)
    assert await agent._run_tool("list_reminders", {}, req, {}, None) == "No reminder is waiting in this chat."
    await agent._run_tool("remind", {"message": "first", "in_minutes": 30}, req, {}, None)
    await agent._run_tool("remind", {"message": "second", "in_minutes": 10}, req, {}, None)
    out = await agent._run_tool("list_reminders", {}, req, {}, None)
    assert out.splitlines()[0].startswith("#2") and "second" in out and "#1" in out   # soonest first
    assert await agent._run_tool("list_reminders", {"cancel_id": 2}, req, {}, None) == "Reminder #2 cancelled."
    assert "second" not in await agent._run_tool("list_reminders", {}, req, {}, None)
    assert "no waiting reminder #2" in await agent._run_tool("list_reminders", {"cancel_id": 2}, req, {}, None)
    # another chat cannot cancel it
    assert "no waiting reminder #1" in await agent._run_tool("list_reminders", {"cancel_id": 1}, request(store, chat="h"), {}, None)


# ── reminders: delivering them ──────────────────────────────────────────────

class FakeBot:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail
        self.messages = SimpleNamespace(send=self._send)

    async def _send(self, message, recipient):
        if self.fail:
            raise ConnectionError("signal-cli down")
        self.sent.append((message.text, recipient))


async def test_due_reminders_are_posted_once():
    store = Store()
    store.add_reminder("g", "u1", "Karim", NOW_MS + 60_000, "Call the plumber", NOW_MS)
    store.add_reminder("g", "u1", "Karim", NOW_MS + 3_600_000, "later", NOW_MS)
    bot, service = FakeBot(), reminders.ReminderService(store)
    assert await service.deliver_due(bot, now_ms=NOW_MS) == 0
    assert await service.deliver_due(bot, now_ms=NOW_MS + 90_000) == 1
    assert bot.sent == [("⏰ Call the plumber", "g")]
    assert await service.deliver_due(bot, now_ms=NOW_MS + 120_000) == 0   # not twice
    assert [r["message"] for r in store.pending_reminders("g")] == ["later"]


async def test_a_reminder_missed_while_offline_is_posted_with_an_apology():
    store = Store()
    store.add_reminder("g", "u1", "Karim", NOW_MS, "Call the plumber", NOW_MS - 1000)
    bot = FakeBot()
    await reminders.ReminderService(store).deliver_due(bot, now_ms=NOW_MS + 3 * 3_600_000)
    assert bot.sent[0][0] == "⏰ Call the plumber (sorry, late: I was offline)"


async def test_a_reminder_is_retried_then_given_up():
    store = Store()
    store.add_reminder("g", "u1", "Karim", NOW_MS, "x", NOW_MS)
    service, bot = reminders.ReminderService(store), FakeBot(fail=True)
    for _ in range(8):
        assert await service.deliver_due(bot, now_ms=NOW_MS + 1000) == 0
    assert store._db.execute("SELECT attempts FROM reminders").fetchone()[0] == 5   # then it stops trying
    bot.fail = False
    assert await service.deliver_due(bot, now_ms=NOW_MS + 1000) == 0


async def test_reminders_survive_a_restart(tmp_path):
    path = tmp_path / "bot.sqlite3"
    store = Store(path)
    store.add_reminder("g", "u1", "Karim", NOW_MS + 1000, "still here", NOW_MS)
    store.close()
    bot = FakeBot()
    assert await reminders.ReminderService(Store(path)).deliver_due(bot, now_ms=NOW_MS + 5000) == 1


async def test_the_service_starts_its_loop_when_the_bot_is_ready():
    store = Store()
    store.add_reminder("g", "u1", "Karim", 1, "now", 0)
    service, bot = reminders.ReminderService(store), FakeBot()
    await service.handle_ready(SimpleNamespace(bot=bot))
    await asyncio.sleep(0.05)
    service._task.cancel()
    assert bot.sent and bot.sent[0][0].startswith("⏰ now")


# ── pick_random ─────────────────────────────────────────────────────────────

def test_pick_random_draws_from_the_choices(monkeypatch):
    monkeypatch.setattr(webtools.random, "choice", lambda items: items[-1])
    assert webtools.pick_random(["pizza", " sushi ", "", "burger"]) == "Picked: burger"


def test_pick_random_rolls_a_die_and_flips_a_coin(monkeypatch):
    monkeypatch.setattr(webtools.random, "randint", lambda a, b: b)
    assert webtools.pick_random(dice=6) == "Rolled a d6: 6"
    monkeypatch.setattr(webtools.random, "choice", lambda items: items[0])
    assert webtools.pick_random() == "Flipped a coin: heads"


@pytest.mark.parametrize("dice", [1, 0 + 5000, "many"])
def test_pick_random_refuses_a_silly_die(dice):
    assert "sides" in webtools.pick_random(dice=dice)


def test_pick_random_really_is_random_enough():
    seen = {webtools.pick_random(["a", "b", "c"]) for _ in range(60)}
    assert seen == {"Picked: a", "Picked: b", "Picked: c"}


async def test_pick_random_through_the_tool_runner():
    out = await webtools.run("pick_random", {"choices": ["x"]}, TZ)
    assert out == "Picked: x"
