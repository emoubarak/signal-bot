"""Answers that are not replies (2026-11-06, local Qwen at Q2): the asker's line copied back with its
transcript header, then the model's reasoning posted as the answer."""
from __future__ import annotations

from llmbot.agent import answer_problem
from tests.conftest import PAUL, Harness, group_message

QUESTION = "jarvis tu crois que la spiritualité c'est se raser le crâne et mendier ?"
ECHO = "[06:38] Paul: " + QUESTION
REASONING = ("I need to handle this carefully. The user is saying I have a bug.\n\n"
             "Let me draft: \"Paul, attends. J'ai buggé.\"")


def test_echo_and_reasoning_are_flagged_replies_are_not():
    assert answer_problem(ECHO, "Jarvis", []) == "echo"
    assert answer_problem(QUESTION, "Jarvis", [QUESTION]) == "echo"
    assert answer_problem(REASONING, "Jarvis", []) == "reasoning"
    assert answer_problem("Okay, so the plan is to answer in French.", "Jarvis", []) == "reasoning"
    assert answer_problem("[06:38] Jarvis: non, le plérôme est partout", "Jarvis", []) is None  # own header: stripped later
    assert answer_problem("Non. Le plérôme est partout, même dans ton crâne rasé 🙄", "Jarvis", [QUESTION]) is None
    assert answer_problem("Let me guess, encore Musk ?", "Jarvis", []) is None  # one line in English is a reply
    assert answer_problem("mdr", "Jarvis", ["mdr"]) is None  # short echoes are fine


async def test_bad_answer_is_asked_again_and_never_posted(harness: Harness):
    harness.llm.script = [ECHO, "Non, le plérôme est partout."]
    await harness.run(group_message(PAUL, QUESTION))
    assert harness.sent_texts() == ["Non, le plérôme est partout."]
    retry = harness.llm.chat_calls[-1]["messages"]
    assert retry[-1]["role"] == "user" and "Not posted" in retry[-1]["content"]
    assert not any(m["role"] == "assistant" and m.get("content") == ECHO for m in retry)


async def test_second_bad_answer_means_silence(harness: Harness):
    harness.llm.script = [ECHO, REASONING]
    await harness.run(group_message(PAUL, QUESTION))
    assert harness.sent_texts() == []
    assert len(harness.llm.chat_calls) == 2
