"""Tool calls written as text (2026-11-06: raw <tool_call> markup posted in the group after memes were refused)."""
from __future__ import annotations

from unittest.mock import AsyncMock

from llmbot.agent import MAX_ROUNDS, NO_FINAL_ANSWER, strip_tool_markup, tool_calls_in_text
from tests.conftest import LEA, PAUL, PNG_1PX, Harness, group_message

RAW = """<tool_call>
<function=search_messages>
<parameter=query>
Youssef gelano
</parameter>
<parameter=limit>
15
</parameter>
</function>
</tool_call>"""


def test_xml_tool_call_is_parsed_with_typed_values():
    [call] = tool_calls_in_text(RAW)
    assert call["function"]["name"] == "search_messages"
    assert call["function"]["arguments"] == '{"query": "Youssef gelano", "limit": 15}'


def test_json_tool_call_and_markup_removal():
    [call] = tool_calls_in_text('<tool_call>{"name": "get_time", "arguments": {"tz": "Asia/Tokyo"}}</tool_call>')
    assert call["function"]["name"] == "get_time"
    assert strip_tool_markup("Voilà. " + RAW) == "Voilà."
    assert strip_tool_markup("<tool_call>\n<function=make_meme>\n<parameter=x>") == ""  # cut mid-call


async def test_tool_call_written_as_text_still_runs(harness: Harness):
    await harness.run(group_message(LEA, "Youssef gelano a dit un truc"))
    harness.llm.script = [RAW, "Il a dit un truc."]
    await harness.run(group_message(PAUL, "jarvis qu'a dit youssef ?"))
    assert harness.sent_texts()[-1] == "Il a dit un truc."
    tool_out = [m for m in harness.llm.chat_calls[-1]["messages"] if m["role"] == "tool"]
    assert tool_out and "gelano" in tool_out[0]["content"]


async def test_markup_never_reaches_the_chat_when_rounds_run_out(harness: Harness):
    harness.llm.script = [{"name": "get_time", "arguments": {"tz": "Europe/Paris"}}] * MAX_ROUNDS + [RAW]
    await harness.run(group_message(PAUL, "jarvis cherche youssef"))
    assert harness.sent_texts() == [NO_FINAL_ANSWER]
    last = harness.llm.chat_calls[-1]["messages"][-1]
    assert last["role"] == "user" and "No tool calls are left" in last["content"]


async def test_recycling_refusal_happens_once_per_reply(harness: Harness, mocker):
    made = mocker.patch("llmbot.webtools.make_meme", new_callable=AsyncMock,
                        side_effect=[(None, "These words come from your own earlier replies, not from the request: x"),
                                     (PNG_1PX, "Meme made")])
    harness.llm.answer = "Sam parle de gnose et d'Andromède"
    await harness.run(group_message(LEA, "jarvis la gnose ?"))
    call = {"name": "make_meme", "arguments": {"template": "drake", "subject": "Sam", "lines": ["gnose", "Andromède"]}}
    harness.llm.script = [call, call, "voilà"]
    await harness.run(group_message(PAUL, "jarvis fais un meme sur sam"))
    first, second = made.call_args_list
    assert first.kwargs["own_text"] and second.kwargs["own_text"] == ""
    assert harness.sent()[-1].base64_attachments
