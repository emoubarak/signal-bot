"""Behaviour fixed after the audit tickets: mentions and quotes (#11), private access (#10), edits (#9), messages
deleted for everyone (#6), rate limits under a burst (#5), images (#3), the local model's turn-taking (#2)."""
from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid

import httpx
import pytest
from signalbot.test_utils.chat_testing import Envelope, _new_timestamp

from llmbot import handler as handler_module
from llmbot import media, openrouter
from llmbot.openrouter import ChatResult, OpenRouter, OpenRouterError
from llmbot.runtime_config import ConfigError, coerce
from llmbot.store import KIND_IMAGE, KIND_VOICE
from tests.conftest import (
    BOT_NUMBER, CHAT_ID, LEA, PAUL, PNG_1PX, Harness, edit_message, group_message, image_attachment, now_ms,
    quote_of, voice_attachment,
)

PAUL_UUID = PAUL[1]


def bot_quote(ts: int, text: str) -> dict:
    return {"id": ts, "author": BOT_NUMBER, "authorNumber": BOT_NUMBER, "authorUuid": str(uuid.uuid4()), "text": text}


def remote_delete(user, target_ts: int) -> Envelope:
    number, uid, name = user
    ts = _new_timestamp()
    env = {"account": BOT_NUMBER, "envelope": {
        "source": number, "sourceNumber": number, "sourceUuid": uid, "sourceName": name, "sourceDevice": 1,
        "timestamp": ts, "serverReceivedTimestamp": now_ms(), "serverDeliveredTimestamp": now_ms(),
        "dataMessage": {"timestamp": ts, "expiresInSeconds": 0, "viewOnce": False,
                        "remoteDelete": {"timestamp": target_ts},
                        "groupInfo": {"groupId": CHAT_ID, "type": "DELIVER", "revision": 1}}}}
    return Envelope(json.dumps(env))


def prompt_text(harness: Harness) -> str:
    """Everything the model was given in the last call."""
    return json.dumps(harness.llm.chat_calls[-1]["messages"], ensure_ascii=False)


# ── #11 mentions, unknown quoted author ─────────────────────────────────────

async def test_a_mention_is_stored_as_the_name_not_an_empty_character(harness: Harness):
    mention = {"name": "Paul", "number": PAUL[0], "uuid": PAUL[1], "start": 0, "length": 1}
    await harness.run(group_message(LEA, "￼ you're late", mentions=[mention]))
    [m] = harness.store.recent(CHAT_ID, 5)
    assert m.text == "@Paul you're late"


async def test_mentions_use_utf16_offsets_and_the_stored_name_when_the_mention_has_none(harness: Harness):
    await harness.run(group_message(PAUL, "hello"))  # the store now knows who this uuid is
    mention = {"name": None, "number": None, "uuid": PAUL_UUID, "start": 3, "length": 1}
    # "😀" is two UTF-16 units: the mention starts at 3, not 2
    await harness.run(group_message(LEA, "😀 ￼ ok", mentions=[mention]))
    assert harness.store.recent(CHAT_ID, 1)[0].text == "😀 @Paul ok"


async def test_a_mention_whose_name_is_a_signal_id_is_written_with_the_name_the_person_writes_under(harness: Harness):
    await harness.run(group_message(PAUL, "hello"))
    # signal-cli puts the number, or the Signal id when the number is hidden, in a mention's name
    mention = {"name": PAUL_UUID, "number": None, "uuid": PAUL_UUID, "start": 0, "length": 1}
    await harness.run(group_message(LEA, "\ufffc ok", mentions=[mention]))
    assert harness.store.recent(CHAT_ID, 1)[0].text == "@Paul ok"
    stranger = {"name": "33333333-3333-3333-3333-333333333333", "number": None,
                "uuid": "33333333-3333-3333-3333-333333333333", "start": 0, "length": 1}
    await harness.run(group_message(LEA, "\ufffc ok", mentions=[stranger]))
    assert harness.store.recent(CHAT_ID, 1)[0].text == "@someone ok"


async def test_a_mention_of_the_bot_is_written_with_its_name(harness: Harness):
    mention = {"name": "x", "number": BOT_NUMBER, "uuid": str(uuid.uuid4()), "start": 0, "length": 1}
    await harness.run(group_message(LEA, "￼ hi", mentions=[mention]))
    assert harness.store.recent(CHAT_ID, 5)[0].text == "@Jarvis hi"


async def test_a_quote_of_an_unknown_message_never_puts_a_number_in_the_prompt(harness: Harness):
    stranger = ("+33699887766", "99999999-9999-9999-9999-999999999999", "x")
    await harness.run(group_message(LEA, "jarvis what does he mean?", quote=quote_of(stranger, 1, "see you at 8")))
    assert "+33699887766" not in prompt_text(harness)
    assert "Someone" in prompt_text(harness)


# ── #10 private access ──────────────────────────────────────────────────────

def test_a_signal_id_is_accepted_in_the_allowed_list():
    assert coerce("allowed_numbers", [PAUL_UUID, "+33612345678"]) == [PAUL_UUID, "+33612345678"]
    with pytest.raises(ConfigError):
        coerce("allowed_numbers", ["not-a-number"])


async def test_a_private_message_from_outside_the_allowed_list_is_not_answered(mocker, tmp_path):
    h = Harness(mocker, tmp_path, allowed_numbers=[LEA[1]])  # allowed by Signal id
    await h.run(group_message(PAUL, "jarvis hi", private=True), group_message(LEA, "jarvis hi", private=True))
    assert len(h.llm.chat_calls) == 1 and len(h.sent()) == 1   # only Lea was answered
    assert h.store.recent(PAUL[1], 5) == []  # not even recorded
    assert len(h.store.recent(LEA[1], 5)) == 2  # Lea's message and the answer


def set_group_members(h: Harness, members: list[str]) -> None:
    """The group list signalbot got from signal-cli names these members."""
    for entry in h.bot.groups._by_internal_id.values():
        entry.members = members


async def test_a_member_of_an_allowed_group_can_talk_in_private(mocker, tmp_path):
    h = Harness(mocker, tmp_path, allowed_groups=[CHAT_ID])
    await h.run(group_message(LEA, "hi"))     # starts the bot and loads the group list
    set_group_members(h, [PAUL[0]])
    await h.run(group_message(PAUL, "jarvis hi", private=True),
                group_message(LEA, "jarvis hi", private=True))   # Lea is not named in the group's members
    assert len(h.llm.chat_calls) == 1
    assert h.store.recent(LEA[1], 5) == []


async def test_someone_removed_from_the_group_loses_private_access_despite_old_messages(mocker, tmp_path):
    h = Harness(mocker, tmp_path, allowed_groups=[CHAT_ID])
    await h.run(group_message(PAUL, "morning everyone"))          # Paul wrote in the group, then left it
    set_group_members(h, [])
    await h.run(group_message(PAUL, "jarvis hi", private=True))
    assert h.llm.chat_calls == []


async def test_a_member_who_hides_their_number_is_known_by_what_they_wrote_in_the_group(mocker, tmp_path):
    h = Harness(mocker, tmp_path, allowed_groups=[CHAT_ID])
    hidden = (None, "33333333-3333-3333-3333-333333333333", "Hidden")
    await h.run(group_message(hidden, "morning"))
    await h.run(group_message(hidden, "jarvis hi", private=True))
    assert len(h.llm.chat_calls) == 1


async def test_the_owner_can_always_talk_in_private(mocker, tmp_path):
    h = Harness(mocker, tmp_path, allowed_groups=[CHAT_ID], owner_number=PAUL[0])
    await h.run(group_message(PAUL, "jarvis hi", private=True))
    assert len(h.llm.chat_calls) == 1


async def test_with_no_list_private_stays_open(harness: Harness):
    await harness.run(group_message(PAUL, "jarvis hi", private=True))
    assert len(harness.llm.chat_calls) == 1


# ── #9 edits ────────────────────────────────────────────────────────────────

async def test_an_edit_adding_the_name_gets_one_answer(harness: Harness):
    ts = _new_timestamp()
    await harness.run(group_message(PAUL, "where do we eat?", ts=ts))
    assert harness.llm.chat_calls == []
    await harness.run(edit_message(PAUL, "Jarvis, where do we eat?", ts))
    assert len(harness.llm.chat_calls) == 1 and len(harness.sent_texts()) == 1
    assert "Jarvis, where do we eat?" in prompt_text(harness)
    await harness.run(edit_message(PAUL, "Jarvis, where do we eat tonight?", ts))   # a second edit: no second answer
    assert len(harness.llm.chat_calls) == 1


async def test_an_edit_of_a_message_already_answered_does_not_answer_again(harness: Harness):
    ts = _new_timestamp()
    await harness.run(group_message(PAUL, "jarvis hi", ts=ts))
    await harness.run(edit_message(PAUL, "jarvis hello", ts))
    assert len(harness.llm.chat_calls) == 1


async def test_editing_a_photo_caption_keeps_its_description(mocker, tmp_path):
    h = Harness(mocker, tmp_path, describe_images=True)
    h.llm.answer = "a cat on a sofa"
    ts = _new_timestamp()
    await h.run(group_message(PAUL, "look", attachments=[image_attachment()], ts=ts))
    assert h.store.recent(CHAT_ID, 1)[0].text == "look [image: a cat on a sofa]"
    await h.run(edit_message(PAUL, "look at this", ts))
    assert h.store.recent(CHAT_ID, 1)[0].text == "look at this [image: a cat on a sofa]"


async def test_editing_a_voice_note_caption_keeps_the_transcription(harness: Harness):
    ts = _new_timestamp()
    await harness.run(group_message(PAUL, "note", attachments=[voice_attachment()], ts=ts))
    [m] = harness.store.recent(CHAT_ID, 1)
    assert m.kind == KIND_VOICE and m.text == "hi this is a voice note - note"
    await harness.run(edit_message(PAUL, "my note", ts))
    assert harness.store.recent(CHAT_ID, 1)[0].text == "hi this is a voice note - my note"


async def test_an_edit_of_a_reply_to_the_bot_does_not_answer_twice(harness: Harness):
    await harness.run(group_message(PAUL, "jarvis hi"))
    bot_ts = harness.bot_timestamps(CHAT_ID)[-1]
    ts = _new_timestamp()
    await harness.run(group_message(LEA, "and then?", quote=bot_quote(bot_ts, "model reply"), ts=ts))
    assert len(harness.llm.chat_calls) == 2           # answered because it replies to the bot
    await harness.run(edit_message(LEA, "and then??", ts))
    assert len(harness.llm.chat_calls) == 2


async def test_an_edit_does_not_regenerate_an_image_answered_with_a_picture(harness: Harness):
    ts = _new_timestamp()
    harness.llm.script = [{"name": "generate_image", "arguments": {"prompt": "a cat"}}, "here"]
    await harness.run(group_message(PAUL, "jarvis make a cat", ts=ts))
    assert len(harness.llm.image_calls) == 1
    await harness.run(edit_message(PAUL, "jarvis make a cat please", ts))
    assert len(harness.llm.image_calls) == 1


async def test_an_edit_that_arrives_late_is_not_answered(harness: Harness):
    ts = _new_timestamp()
    await harness.run(group_message(PAUL, "where do we eat?", ts=ts))
    old = now_ms() - 3_600_000
    await harness.run(edit_message(PAUL, "Jarvis, where do we eat?", ts, server_ts=old))
    assert harness.llm.chat_calls == []
    assert harness.store.get(CHAT_ID, ts).text == "Jarvis, where do we eat?"   # still kept


async def test_an_edit_by_someone_else_changes_nothing(harness: Harness):
    ts = _new_timestamp()
    await harness.run(group_message(PAUL, "hello", ts=ts))
    await harness.run(edit_message(LEA, "jarvis hacked", ts))
    assert harness.store.recent(CHAT_ID, 1)[0].text == "hello" and harness.llm.chat_calls == []


# ── #6 deleted for everyone ─────────────────────────────────────────────────

async def test_a_message_deleted_for_everyone_leaves_the_history_and_its_files(harness: Harness):
    ts = _new_timestamp()
    await harness.run(group_message(PAUL, "secret pizza plan", attachments=[image_attachment()], ts=ts),
                      group_message(LEA, "nice"))
    [att] = harness.store.get(CHAT_ID, ts).attachments
    assert media.Path(att["path"]).exists()
    await harness.run(remote_delete(PAUL, ts))
    assert harness.store.get(CHAT_ID, ts) is None
    assert [m.text for m in harness.store.recent(CHAT_ID, 10)] == ["nice"]
    assert harness.store.search(CHAT_ID, "pizza") == []
    assert not media.Path(att["path"]).exists()


async def test_only_the_author_can_delete_a_message(harness: Harness):
    ts = _new_timestamp()
    await harness.run(group_message(PAUL, "keep me", ts=ts))
    await harness.run(remote_delete(LEA, ts))
    assert harness.store.get(CHAT_ID, ts) is not None


async def test_a_deleted_message_is_not_shown_to_the_model(harness: Harness):
    ts = _new_timestamp()
    await harness.run(group_message(PAUL, "the code is 4242", ts=ts))
    await harness.run(remote_delete(PAUL, ts))
    await harness.run(group_message(LEA, "jarvis what was said before?"))
    assert "4242" not in prompt_text(harness)


# ── #5 rate limits ──────────────────────────────────────────────────────────

async def test_a_burst_is_limited_and_slow_down_is_said_once(mocker, tmp_path):
    h = Harness(mocker, tmp_path, rate_limit_per_hour=2)
    await h.run(*[group_message(PAUL, f"jarvis question {i}") for i in range(5)], drain=False)
    await h.handler.drain()
    texts = h.sent_texts()
    assert texts.count("model reply") == 2
    assert len([t for t in texts if t.startswith("Slow down")]) == 1
    assert len(h.llm.chat_calls) == 2


async def test_a_failed_request_counts_against_the_limit(mocker, tmp_path):
    h = Harness(mocker, tmp_path, rate_limit_per_hour=1)

    async def boom(**kw):
        raise OpenRouterError("HTTP 500: down", 500)
    h.llm.chat = boom
    await h.run(group_message(PAUL, "jarvis one"), group_message(PAUL, "jarvis two"))
    texts = h.sent_texts()
    assert texts[0].startswith("The model did not answer") and texts[1].startswith("Slow down")


async def test_requests_of_one_person_do_not_overlap(mocker, tmp_path):
    h = Harness(mocker, tmp_path)
    running, peak = 0, 0
    original = h.llm.chat

    async def slow(**kw):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        return await original(**kw)
    h.llm.chat = slow
    await h.run(*[group_message(PAUL, f"jarvis {i}") for i in range(3)], drain=False)
    await h.handler.drain()
    assert peak == 1 and len(h.sent_texts()) == 3


# ── #3 images ───────────────────────────────────────────────────────────────

def tiff_attachment() -> dict:
    return {"id": "scan.tiff", "contentType": "image/tiff", "isVoiceNote": False, "size": len(PNG_1PX)}


async def test_an_unreadable_image_is_replaced_by_a_note_and_the_answer_still_comes(harness: Harness):
    await harness.run(group_message(PAUL, "jarvis what is this?", attachments=[tiff_attachment()]))
    assert harness.sent_texts() == ["model reply"]
    sent_parts = harness.llm.chat_calls[-1]["messages"][-1]["content"]
    assert not any(p.get("type") == "image_url" for p in sent_parts)
    assert any(p.get("text") == "[unreadable image]" for p in sent_parts)


def test_image_for_model_refuses_what_it_cannot_convert(mocker):
    mocker.patch("llmbot.media.ffmpeg_available", return_value=False)
    assert media.image_for_model(b"x", "image/tiff") is None
    assert media.image_for_model(b"x", "image/svg+xml") is None
    assert media.image_for_model(b"png", "image/png") == (b"png", "image/png")
    big = b"x" * (media.MAX_IMAGE_BYTES + 1)
    assert media.image_for_model(big, "image/jpeg") == (big, "image/jpeg")  # accepted type: sent as it is


async def test_at_most_the_latest_images_reach_the_model(harness: Harness):
    from llmbot import agent as agent_module
    for _ in range(agent_module.MAX_IMAGES + 3):
        await harness.run(group_message(PAUL, "pic", attachments=[image_attachment()]))
    # one thread of replies would carry them all: here the cap is checked directly
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "x"},
        *[{"type": agent_module.PENDING_IMAGE, "path": m.attachments[0]["path"], "content_type": "image/png"}
          for m in harness.store.recent(CHAT_ID, 20)]]}]
    await harness.handler.agent._load_images(messages)
    parts = messages[0]["content"]
    assert sum(p["type"] == "image_url" for p in parts) == agent_module.MAX_IMAGES
    assert sum(p.get("text") == "[older image not shown]" for p in parts) == 3


async def test_a_slow_conversion_does_not_freeze_the_event_loop(mocker):
    def slow(data, content_type):
        time.sleep(0.3)
        return data, "image/png"
    mocker.patch("llmbot.media.image_for_model", slow)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1
    task = asyncio.create_task(ticker())
    await media.load_image(PNG_1PX, "image/png")
    task.cancel()
    assert ticks >= 10  # the loop kept running during the 0.3 s conversion


async def test_a_message_recorded_after_a_photo_is_not_delayed_by_its_description(mocker, tmp_path):
    h = Harness(mocker, tmp_path, describe_images=True)
    gate = asyncio.Event()
    original = h.llm.chat

    async def slow_chat(**kw):
        await gate.wait()
        return await original(**kw)
    h.llm.chat = slow_chat
    h.llm.answer = "a red car"
    await h.run(group_message(PAUL, "my car", attachments=[image_attachment()]),
                group_message(LEA, "nice one"), drain=False)
    # both recorded while the description is still waiting
    assert [m.text for m in h.store.recent(CHAT_ID, 5)] == ["my car", "nice one"]
    gate.set()
    await h.handler.drain()
    assert h.store.recent(CHAT_ID, 5)[0].text == "my car [image: a red car]"


# ── #2 the local model takes one request at a time ──────────────────────────

def local_llm(handler_fn) -> OpenRouter:
    client = httpx.AsyncClient(base_url="http://local/v1", transport=httpx.MockTransport(handler_fn))
    return OpenRouter("", base_url="http://local/v1", local=True, client=client)


def ok_body() -> dict:
    return {"model": "m", "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}


async def test_two_local_requests_take_turns_and_the_second_is_not_timed_out_by_the_first(monkeypatch):
    monkeypatch.setattr(openrouter, "LOCAL_CHAT_DEADLINE", 0.5)
    running, peak = 0, 0

    async def server(request: httpx.Request) -> httpx.Response:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.3)
        running -= 1
        return httpx.Response(200, json=ok_body())

    llm = local_llm(server)
    started = time.monotonic()
    a, b = await asyncio.gather(llm.chat(model="m", messages=[]), llm.chat(model="m", messages=[]))
    assert (a.text, b.text) == ("hi", "hi")
    assert peak == 1                                # one at a time
    assert time.monotonic() - started > 0.55        # longer than the deadline in all, yet neither timed out


async def test_a_503_from_the_local_server_is_retried_once(monkeypatch):
    monkeypatch.setattr(openrouter, "LOCAL_RETRY_PAUSE", 0.01)
    calls = []

    def server(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "starting"}}) if len(calls) == 1 else \
            httpx.Response(200, json=ok_body())

    assert (await local_llm(server).chat(model="m", messages=[])).text == "hi"
    assert len(calls) == 2


async def test_a_second_503_is_reported(monkeypatch):
    monkeypatch.setattr(openrouter, "LOCAL_RETRY_PAUSE", 0.01)
    calls = []

    def server(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "starting"}})

    with pytest.raises(OpenRouterError):
        await local_llm(server).chat(model="m", messages=[])
    assert len(calls) == 2


async def test_the_group_is_told_once_when_the_local_model_is_slow(mocker, tmp_path, monkeypatch):
    monkeypatch.setattr(handler_module, "LOADING_NOTICE_SECONDS", 0.05)
    h = Harness(mocker, tmp_path, local_base_url="http://local/v1")
    original = h.llm.chat

    async def slow(**kw):
        await asyncio.sleep(0.2)
        return await original(**kw)
    h.llm.chat = slow
    await h.run(group_message(PAUL, "jarvis hi"))
    assert h.sent_texts() == [handler_module.LOADING_NOTICE, "model reply"]
    # the notice is not one of the bot's answers
    assert [m.text for m in h.store.recent(CHAT_ID, 5)] == ["jarvis hi", "model reply"]


async def test_a_fast_local_answer_gets_no_notice(mocker, tmp_path, monkeypatch):
    monkeypatch.setattr(handler_module, "LOADING_NOTICE_SECONDS", 0.2)
    h = Harness(mocker, tmp_path, local_base_url="http://local/v1")
    await h.run(group_message(PAUL, "jarvis hi"))
    assert h.sent_texts() == ["model reply"]


async def test_the_notice_is_not_sent_when_the_model_answered_and_the_rest_is_slow(mocker, tmp_path, monkeypatch):
    monkeypatch.setattr(handler_module, "LOADING_NOTICE_SECONDS", 0.05)
    h = Harness(mocker, tmp_path, local_base_url="http://local/v1")
    h.llm.script = [{"name": "get_time", "arguments": {"timezone": "Europe/Paris"}}, "it is late"]
    from llmbot import webtools
    original = webtools.run

    async def slow_tool(*a, **kw):
        await asyncio.sleep(0.2)   # the tool is slow, not the model
        return await original(*a, **kw)
    monkeypatch.setattr(webtools, "run", slow_tool)
    await h.run(group_message(PAUL, "jarvis what time is it?"))
    assert h.sent_texts() == ["it is late"]


async def test_reminders_the_old_version_gave_up_on_stay_given_up(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.executescript("""CREATE TABLE reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL,
        author_id TEXT NOT NULL, author_name TEXT NOT NULL, due_ts INTEGER NOT NULL, message TEXT NOT NULL,
        created_ts INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, done INTEGER NOT NULL DEFAULT 0);
        INSERT INTO reminders (chat_id, author_id, author_name, due_ts, message, created_ts, attempts)
        VALUES ('g','u','N',1,'dead',1,5), ('g','u','N',1,'alive',1,2);""")
    db.commit()
    db.close()
    from llmbot.store import Store
    store = Store(path)
    assert [r["message"] for r in store.due_reminders(10)] == ["alive"]
    assert [r["message"] for r in store.pending_reminders("g")] == ["alive"]
