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
