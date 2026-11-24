"""Who "tu" and "je" are (2026-11-24): told "et ta mère ?", the bot answered about somebody else's mother
("sa mère c'est la vraie stoïcienne"), and talked about the author to the author ("le GPU de Mehdiurge")."""
from __future__ import annotations

from tests.conftest import CHAT_ID, LEA, Harness, bot_quote, group_message


def _last_user_text(harness: Harness) -> str:
    last = [m for m in harness.llm.chat_calls[-1]["messages"] if m["role"] == "user"][-1]
    return last["content"] if isinstance(last["content"], str) else last["content"][0]["text"]


async def test_the_message_answered_says_who_tu_and_je_are(harness: Harness):
    await harness.run(group_message(LEA, "jarvis et ta mère elle fait quoi"))
    text = _last_user_text(harness)
    assert text.startswith("Lea: jarvis et ta mère")
    assert "Written by Lea" in text and "means you, Jarvis" in text and '"ma mère", never "sa mère"' in text


async def test_a_reply_to_the_bot_names_it_as_itself(harness: Harness):
    harness.llm.answer = "Salut"
    await harness.run(group_message(LEA, "jarvis yo"))
    bot_ts = harness.bot_timestamps(CHAT_ID)[-1]
    await harness.run(group_message(LEA, "et ta mère", quote=bot_quote(bot_ts, "Salut")))
    assert _last_user_text(harness).startswith("Lea (replying to Jarvis (you)): et ta mère")


def _system(harness: Harness) -> str:
    return harness.llm.chat_calls[-1]["messages"][0]["content"]


async def test_the_prompt_lists_members_with_their_old_display_names_and_the_note(harness: Harness):
    from dataclasses import replace
    old_name = (LEA[0], LEA[1], "Lili")
    await harness.run(group_message(old_name, "salut"))
    harness.handler.agent.settings = replace(harness.handler.agent.settings, people="OT = Tronji = Othman")
    await harness.run(group_message(LEA, "jarvis qui est Tronji"))
    system = _system(harness)
    assert "Lea (earlier display names: Lili)" in system
    assert "OT = Tronji = Othman" in system and "Jarvis (earlier" not in system


async def test_remember_who_merges_into_the_note_and_is_saved(harness: Harness):
    from llmbot import runtime_config as rc
    harness.llm.script = [{"name": "remember_who", "arguments": {"names": ["OT", "Othman"]}}, "noté",
                          {"name": "remember_who", "arguments": {"names": ["Tronji", "OT"]}}, "noté",
                          {"name": "remember_who", "arguments": {"names": ["othman", "OT"]}}, "déjà su"]
    await harness.run(group_message(LEA, "jarvis OT c'est Othman"))
    await harness.run(group_message(LEA, "jarvis Tronji c'est OT aussi"))
    await harness.run(group_message(LEA, "jarvis othman c'est OT"))
    assert rc.load(harness.store)["people"] == "OT = Othman = Tronji"
    assert harness.handler.agent.settings.people == "OT = Othman = Tronji"
    tool_out = [m for m in harness.llm.chat_calls[-1]["messages"] if m["role"] == "tool"][-1]["content"]
    assert tool_out.startswith("Already known")


def test_remember_who_keeps_what_someone_is():
    from zoneinfo import ZoneInfo
    from llmbot.agent import Agent
    from llmbot.config import Settings
    from llmbot.store import Store
    store = Store(":memory:")
    a = Agent(Settings(signal_service="x", phone_number="+33600000000", local_base_url="http://x"),
              store, None, None, ZoneInfo("UTC"))
    assert a._remember_who(["Kader"], "Badr's cousin, not in the group").startswith("Saved")
    a._remember_who(["kader", "Kad"])
    assert a._remember_who(["Kad"], "member of the group").startswith("Already known")
    assert a._remember_who(["Kad", "Kader"], "member of the group").startswith("Already known")
    assert a.settings.people == "Kader = Kad (Badr's cousin, not in the group)"
    assert a._remember_who(["Kader"]).startswith("Give at least two")
