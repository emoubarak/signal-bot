"""`/clear`: the owner wipes what the model can see in a chat, without deleting anything."""
from zoneinfo import ZoneInfo

import pytest

from llmbot.agent import Agent, AgentRequest
from llmbot.config import Settings
from llmbot.personas import Registry
from llmbot.store import Store, StoredMessage

OWNER = "+33628508830"
OTHER = "+33611111111"


def settings(**overrides) -> Settings:
    base = {"signal_service": "127.0.0.1:8080", "phone_number": "+33745154734",
            "bot_name": "Mop", "image_model": "", "transcribe_model": "",
            "local_base_url": "http://x:8089/v1", "owner_number": OWNER}
    return Settings(**{**base, **overrides})


def msg(ts, author, text, kind="text"):
    return StoredMessage(chat_id="chat", ts=ts, author_id="id" + author, author_name=author, kind=kind, text=text)


# ── store ───────────────────────────────────────────────────────────────────

def test_clear_context_hides_earlier_messages_without_deleting_them():
    store = Store(":memory:")
    for i in range(5):
        store.add(msg(1000 + i, "Paul", f"message {i}"))

    hidden = store.clear_context("chat", 1003)

    assert hidden == 3                      # 1000, 1001, 1002 are now out of view
    assert store.get("chat", 1000) is not None   # but still in the database
    assert store.context_start("chat") == 1003


def test_context_start_is_none_before_any_clear():
    store = Store(":memory:")
    store.add(msg(1000, "Paul", "salut"))
    assert store.context_start("chat") is None


def test_clear_is_per_chat():
    store = Store(":memory:")
    store.clear_context("group-a", 2000)
    assert store.context_start("group-a") == 2000
    assert store.context_start("group-b") is None


def test_a_second_clear_moves_the_floor():
    store = Store(":memory:")
    store.clear_context("chat", 1000)
    store.clear_context("chat", 5000)
    assert store.context_start("chat") == 5000


def test_clear_survives_a_reopened_store(tmp_path):
    path = tmp_path / "bot.sqlite3"
    store = Store(path)
    store.clear_context("chat", 1234)
    store.close()

    assert Store(path).context_start("chat") == 1234


# ── what the model is shown ─────────────────────────────────────────────────

def _agent(store, **overrides) -> Agent:
    return Agent(settings(**overrides), store, llm=None, registry=Registry.load(None),
                 tz=ZoneInfo("Europe/Paris"))


def test_window_before_the_clear_is_not_sent_to_the_model():
    store = Store(":memory:")
    agent = _agent(store)
    old = [msg(1000, "Paul", "on parle de politique"), msg(1001, "Lea", "ouais")]
    new = [msg(2000, "Paul", "nouveau sujet")]
    req = AgentRequest(chat_id="chat", chat_name="Groupe", user_id="idPaul",
                       thread=[msg(3000, "Paul", "Mop, tu en penses quoi ?")],
                       window=old + new, quoted=None, images_remaining=0, context_start=2000)

    messages = agent._build_messages(req, {})
    flat = "\n".join(str(m["content"]) for m in messages)

    assert "nouveau sujet" in flat
    assert "politique" not in flat          # the biased old context is gone


def test_the_clear_itself_stays_visible():
    """The /clear message is the new floor: the model still sees it and what follows."""
    store = Store(":memory:")
    agent = _agent(store)
    req = AgentRequest(chat_id="chat", chat_name=None, user_id="idMehdi",
                       thread=[msg(5000, "Mehdi", "Mop ca va ?")],
                       window=[msg(4000, "Mehdi", "/clear"), msg(4500, "Paul", "avant/apres")],
                       quoted=None, images_remaining=0, context_start=4000)

    flat = "\n".join(str(m["content"]) for m in agent._build_messages(req, {}))

    assert "avant/apres" in flat            # said after the clear: still visible
    assert "Mop ca va ?" in flat


async def test_get_history_cannot_reach_past_the_clear():
    store = Store(":memory:")
    for i in range(5):
        store.add(msg(1000 + i, "Paul", f"ancien {i}"))
    store.add(msg(3000, "Lea", "recent"))
    store.clear_context("chat", 2000)

    agent = _agent(store)
    req = AgentRequest(chat_id="chat", chat_name=None, user_id="idLea",
                       thread=[msg(4000, "Lea", "Mop résume")], window=[], quoted=None,
                       images_remaining=0, context_start=2000)

    out = await agent._run_tool("get_history", {"count": 100}, req, {}, None)

    assert "recent" in out
    assert "ancien" not in out              # the model cannot dig the old context back up


async def test_get_history_says_so_when_there_is_nothing_since_the_clear():
    store = Store(":memory:")
    store.add(msg(1000, "Paul", "vieux message"))
    store.clear_context("chat", 2000)

    agent = _agent(store)
    req = AgentRequest(chat_id="chat", chat_name=None, user_id="idPaul",
                       thread=[msg(3000, "Paul", "Mop résume hier")], window=[], quoted=None,
                       images_remaining=0, context_start=2000)

    out = await agent._run_tool("get_history", {"count": 100, "since_hours": 48}, req, {}, None)

    assert "clear" in out.lower()
    assert "vieux message" not in out


def test_the_model_is_told_the_context_was_cleared():
    store = Store(":memory:")
    agent = _agent(store)
    req = AgentRequest(chat_id="chat", chat_name=None, user_id="idPaul",
                       thread=[msg(3000, "Paul", "Mop résume")], window=[], quoted=None,
                       images_remaining=0, context_start=2000)
    system = agent._build_messages(req, {})[0]["content"]
    assert "contexte a été vidé" in system


def test_without_a_clear_the_model_sees_everything():
    store = Store(":memory:")
    agent = _agent(store)
    req = AgentRequest(chat_id="chat", chat_name=None, user_id="idPaul",
                       thread=[msg(3000, "Paul", "Mop résume")],
                       window=[msg(1000, "Paul", "vieux sujet")], quoted=None, images_remaining=0)
    flat = "\n".join(str(m["content"]) for m in agent._build_messages(req, {}))
    assert "vieux sujet" in flat


# ── the command, through the handler ────────────────────────────────────────

from tests.conftest import CHAT_ID, LEA, PAUL, Harness, group_message  # noqa: E402


@pytest.fixture
def owner_harness(mocker, tmp_path):
    """The default harness has no OWNER_NUMBER: /clear is for the owner only."""
    return Harness(mocker, tmp_path, owner_number=OWNER)


async def test_owner_clears_the_context_in_the_group(owner_harness: Harness):
    harness = owner_harness
    owner = (OWNER, "owner-uuid", "Mehdi")
    await harness.run(group_message(PAUL, "on parle de politique"), group_message(owner, "/clear"))

    sent = harness.sent_texts()
    assert any("de zéro" in t for t in sent)          # the bot confirms in the group
    assert harness.store.context_start(CHAT_ID) is not None
    # the model was never asked to answer the command itself
    assert harness.llm.chat_calls == []


async def test_another_person_cannot_clear_the_context(owner_harness: Harness):
    harness = owner_harness
    await harness.run(group_message(PAUL, "/clear"))

    assert harness.store.context_start(CHAT_ID) is None
    assert harness.sent_texts() == []                 # and the bot stays quiet about it


async def test_clear_must_be_the_whole_command(owner_harness: Harness):
    harness = owner_harness
    owner = (OWNER, "owner-uuid", "Mehdi")
    await harness.run(group_message(owner, "/clearance des travaux commence demain"))

    assert harness.store.context_start(CHAT_ID) is None


async def test_clear_without_an_owner_configured_does_nothing(harness: Harness):
    # le harnais par defaut n'a pas de OWNER_NUMBER
    owner = (OWNER, "owner-uuid", "Mehdi")
    await harness.run(group_message(owner, "/clear"))

    assert harness.store.context_start(CHAT_ID) is None


@pytest.mark.parametrize("spelling", ["/clear", "!clear", "/clear!", "/Clear", "  /clear  "])
async def test_clear_accepts_the_usual_spellings(owner_harness: Harness, spelling: str):
    owner = (OWNER, "owner-uuid", "Mehdi")
    await owner_harness.run(group_message(owner, spelling))
    assert owner_harness.store.context_start(CHAT_ID) is not None, spelling


@pytest.mark.parametrize("text", ["/clearance des travaux", "/clearer la table", "/clearing"])
async def test_clear_rejects_words_that_only_start_with_it(owner_harness: Harness, text: str):
    owner = (OWNER, "owner-uuid", "Mehdi")
    await owner_harness.run(group_message(owner, text))
    assert owner_harness.store.context_start(CHAT_ID) is None, text
