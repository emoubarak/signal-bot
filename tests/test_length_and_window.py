"""Reply length budget and the conversation window (2026-11-07): replies had a median of 75 words against 9
for a person's turn, and the bot's own earlier replies were 74 % of the context it was shown."""
from __future__ import annotations

from types import SimpleNamespace

from llmbot.agent import (BUDGETS, CUT, GAP_MS, MAX_ROUNDS, SHORTEN_NOTE, TOKEN_CAPS, WINDOW_LINES, Agent,
                          AgentRequest, answer_problem, cut_to_words, length_class)
from llmbot.store import KIND_BOT, KIND_IMAGE, StoredMessage
from tests.conftest import LEA, PAUL, Harness, group_message

LONG_REPLY = " ".join(["Le démiurge a encore frappé."] * 20)   # 100 words, five words a sentence
SHORT_REPLY = "Le démiurge t'a eu, frère."
AGENT = Agent.__new__(Agent)
AGENT.settings = SimpleNamespace(phone_number="+bot")


def msg(ts, who="Paul", text="salut", bot=False):
    return StoredMessage("c", ts, "+bot" if bot else "u-" + who, who, KIND_BOT if bot else "text", text)


def test_length_class():
    assert length_class("mop tu es là ?") == "short"
    assert length_class("Mop roast mon ami") == "medium"
    assert length_class("Mop parle nous comme un rabbin") == "medium"
    assert length_class("Mop pourquoi l'angola a une machette") == "medium"
    assert length_class('mop fait des jeux de mot avec le nom "legmas"') == "medium"
    assert length_class("Mop résume") == "long"
    assert length_class("mop dresse un profiling de chaque personne") == "long"
    assert length_class("mop donne nous des conseils") == "long"
    assert length_class("Mop fait un classement des religions par niveau d'abbération") == "long"
    assert length_class("mop top 5 des pires films") == "long"
    assert length_class("mop c'est qui", with_image=True) == "medium"
    assert length_class("mop t'as vu ça https://tiktok.com/x") == "short"  # read_page makes it long, not the link
    for not_long in ("contenu explicite", "on se tutoie", "présumé innocent", "t'as vu son profil", "conseil municipal"):
        assert length_class(not_long) == "short", not_long
    for not_medium in ("je veux un croissant", "j'y songe", "qui est chaud ce soir", "je pense oui"):
        assert length_class(not_medium) == "short", not_medium


def test_cut_keeps_whole_sentences():
    assert cut_to_words(SHORT_REPLY, 30) == SHORT_REPLY
    assert cut_to_words(LONG_REPLY, 12) == "Le démiurge a encore frappé. Le démiurge a encore frappé."
    assert cut_to_words("une phrase de sept mots sans fin", 3) == "une phrase de…"
    assert cut_to_words("Fini. Pas fin", 30, truncated=True) == "Fini."


def test_window_people_only_since_the_last_pause():
    h = 3600 * 1000
    window = ([msg(0, text="vieux sujet")]
              + [msg(3 * h + i, "Lea" if i % 2 else "Paul", f"m{i}") for i in range(20)]
              + [msg(3 * h + 30, "Mop", "ancienne blague", bot=True),
                 StoredMessage("c", 3 * h + 40, "+bot", "Mop", KIND_IMAGE, "[gif] mdr")])  # a GIF the bot sent
    thread = [msg(3 * h + 60, text="mop ?")]
    kept = AGENT._window(AgentRequest("c", None, "u", thread, window, None, 0), thread, None, None)
    people = [m for m in kept if m.author_id != "+bot"]
    assert len(people) == WINDOW_LINES and kept[-1].text == "[gif] mdr"   # the bot's own replies are kept, not counted
    assert [m.text for m in kept if m.author_id == "+bot"] == ["ancienne blague", "[gif] mdr"]
    assert "vieux sujet" not in [m.text for m in kept]
    # a pause longer than GAP_MS just before the call: nothing from before it
    late = [msg(3 * h + GAP_MS + 1000, text="mop ?")]
    assert AGENT._window(AgentRequest("c", None, "u", late, window, None, 0), late, None, None) == []


def test_echo_in_the_new_line_form():
    said = "salut tout le monde ça va bien ce soir"
    assert answer_problem("Alice: " + said, "Jarvis", [said], ("Alice",)) == "echo"
    # a profile of each person opens with a name too (2026-11-07, silenced by a plainer rule)
    assert answer_problem("Alice: la reine du shitpost à 3h du matin.\nBob: le banquier.", "Jarvis", [said],
                          ("Alice", "Bob")) is None
    assert answer_problem("Bref: on mange où ?", "Jarvis", [], ("Alice",)) is None
    assert answer_problem("Alice, t'as raison", "Jarvis", [], ("Alice",)) is None


async def test_lines_have_no_time_and_the_budget_is_given(harness: Harness):
    await harness.run(group_message(LEA, "on mange où ce soir ?"))
    await harness.run(group_message(PAUL, "jarvis t'en dis quoi"))
    system, user = harness.llm.last_system(), harness.llm.last_user_text()
    assert "Lea: on mange où ce soir ?" in system and "] Lea:" not in system
    assert user.startswith("Paul: jarvis") and f"at most {BUDGETS['short']} words" in user
    assert harness.llm.chat_calls[-1]["max_tokens"] == TOKEN_CAPS["short"]


async def test_too_long_is_rewritten_once_without_tools(harness: Harness):
    harness.llm.script = [LONG_REPLY, SHORT_REPLY]
    await harness.run(group_message(PAUL, "jarvis t'es là ?"))
    assert harness.sent_texts() == [SHORT_REPLY]
    second = harness.llm.chat_calls[-1]
    assert second["tools"] is None
    assert second["messages"][-2] == {"role": "assistant", "content": LONG_REPLY}
    assert second["messages"][-1]["content"] == SHORTEN_NOTE.format(words=100, budget=BUDGETS["short"])


async def test_still_too_long_is_cut_after_a_sentence(harness: Harness):
    harness.llm.script = [LONG_REPLY, LONG_REPLY]
    await harness.run(group_message(PAUL, "jarvis t'es là ?"))
    [sent] = harness.sent_texts()
    assert len(sent.split()) <= BUDGETS["short"] * CUT and sent.endswith(".")
    assert len(harness.llm.chat_calls) == 2


async def test_a_summary_may_be_long_and_the_model_is_told(harness: Harness):
    harness.llm.script = [{"name": "get_history", "arguments": {"count": 50}}, LONG_REPLY]
    await harness.run(group_message(PAUL, "jarvis c'était quoi le délire hier"))
    assert harness.sent_texts() == [LONG_REPLY]  # 100 words, under the long budget once history was read
    last = harness.llm.chat_calls[-1]
    assert last["max_tokens"] == TOKEN_CAPS["long"]
    tool_msg = [m for m in last["messages"] if m["role"] == "tool"][-1]
    assert tool_msg["content"].endswith(f"(Your reply: at most {BUDGETS['long']} words.)")


async def test_one_retry_in_all(harness: Harness):
    harness.llm.script = ["[06:38] Paul: jarvis t'es là ?", LONG_REPLY]
    await harness.run(group_message(PAUL, "jarvis t'es là ?"))
    assert len(harness.llm.chat_calls) == 2   # the echo used the retry: the long one is cut, not asked again
    [sent] = harness.sent_texts()
    assert len(sent.split()) <= BUDGETS["short"] * CUT


async def test_on_the_last_round_a_long_answer_is_cut_not_sent_back(harness: Harness):
    harness.llm.script = ([{"name": "get_time", "arguments": {"tz": "Europe/Paris"}}] * (MAX_ROUNDS - 1)
                          + [" ".join(["Une phrase de cinq mots."] * 120)])
    await harness.run(group_message(PAUL, "jarvis quelle heure"))
    [sent] = harness.sent_texts()
    assert len(harness.llm.chat_calls) == MAX_ROUNDS
    assert len(sent.split()) <= BUDGETS["short"] * CUT


async def test_no_tool_written_as_text_runs_after_a_shorten_note(harness: Harness):
    harness.llm.script = [LONG_REPLY, '<tool_call>{"name": "get_time", "arguments": {"tz": "Asia/Tokyo"}}</tool_call>']
    await harness.run(group_message(PAUL, "jarvis t'es là ?"))
    assert len(harness.llm.chat_calls) == 2   # the written call did not run (no third call)
    [sent] = harness.sent_texts()
    assert sent.startswith("Le démiurge") and len(sent.split()) <= BUDGETS["short"] * CUT  # the draft, cut


async def test_em_dashes_never_posted(harness: Harness):
    harness.llm.script = ["Le Plerome, c'est le spirituel pur — pas de Tesla.\n— et voilà"]
    await harness.run(group_message(PAUL, "jarvis musk ?"))
    assert harness.sent_texts() == ["Le Plerome, c'est le spirituel pur, pas de Tesla.\n- et voilà"]


async def test_history_tool_cuts_the_bots_own_replies(harness: Harness):
    harness.llm.answer = " ".join(f"mot{i}" for i in range(40))
    await harness.run(group_message(PAUL, "jarvis dis un truc"))
    harness.llm.script = [{"name": "get_history", "arguments": {"count": 10}}, "ok"]
    await harness.run(group_message(LEA, "jarvis résume"))
    out = [m for m in harness.llm.chat_calls[-1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "Paul: jarvis dis un truc" in out and "Jarvis (you) (replying to Paul): mot0" in out
    assert "mot24…" in out and "mot25" not in out
