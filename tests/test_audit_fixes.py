"""Fixes from the 2026-11-06 audit: duplicates, "@name", DST, bad tool arguments, admin behind a proxy."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from signalbot.test_utils.chat_testing import _new_timestamp

from llmbot.reminders import parse_when
from tests.conftest import LEA, PAUL, Harness, group_message
from tests.test_admin import TOKEN, admin, client  # noqa: F401  (fixture)


async def test_a_message_delivered_twice_is_answered_once(harness: Harness):
    ts = _new_timestamp()
    await harness.run(group_message(PAUL, "jarvis hello", ts=ts))
    await harness.run(group_message(PAUL, "jarvis hello", ts=ts))
    assert harness.sent_texts() == ["model reply"] and len(harness.llm.chat_calls) == 1


async def test_same_timestamp_from_someone_else_is_not_a_duplicate(harness: Harness):
    ts = _new_timestamp()
    await harness.run(group_message(PAUL, "jarvis hello", ts=ts), group_message(LEA, "jarvis hi", ts=ts))
    assert len(harness.sent()) == 2


async def test_at_name_typed_by_hand_calls_the_bot(harness: Harness):
    await harness.run(group_message(PAUL, "@Jarvis t'es là ?"))
    assert harness.sent_texts() == ["model reply"]
    await harness.run(group_message(LEA, "my jarvis@home"))
    assert len(harness.sent()) == 1


def test_in_minutes_across_the_autumn_clock_change_is_real_time():
    tz = ZoneInfo("Europe/Paris")
    now = datetime(2026, 10, 25, 2, 30, tzinfo=tz)  # first 02:30 (CEST), one hour before the change back
    due, err = parse_when(None, 60, tz, now)
    assert not err and (due - int(now.timestamp() * 1000)) == 60 * 60_000


async def test_bad_tool_arguments_go_back_to_the_model(harness: Harness):
    harness.llm.script = [{"name": "get_history", "arguments": {"count": "all"}}, "Here you go."]
    await harness.run(group_message(PAUL, "jarvis summarize everything"))
    assert harness.sent_texts() == ["Here you go."]
    tool_out = [m["content"] for m in harness.llm.chat_calls[-1]["messages"] if m["role"] == "tool"][0]
    assert "Invalid arguments" in tool_out


async def test_tool_arguments_that_are_not_an_object_are_ignored(harness: Harness):
    harness.llm.script = [{"name": "get_history", "arguments": [1, 2]}, "ok"]
    await harness.run(group_message(PAUL, "jarvis recap"))
    assert harness.sent_texts() == ["ok"]


def test_lockout_behind_tailscale_serve_is_per_visitor(admin):  # noqa: F811
    a = httpx.Client(base_url=admin.base, headers={"X-Admin": "1", "X-Forwarded-For": "100.64.0.1"})
    b = httpx.Client(base_url=admin.base, headers={"X-Admin": "1", "X-Forwarded-For": "100.64.0.2"})
    for _ in range(5):
        assert a.post("/api/login", json={"token": "nope"}).status_code == 401
    assert a.post("/api/login", json={"token": TOKEN}).status_code == 429
    assert b.post("/api/login", json={"token": TOKEN}).status_code == 200  # someone else still gets in


def test_cookie_is_secure_only_over_https(admin):  # noqa: F811
    plain = httpx.post(admin.base + "/api/login", json={"token": TOKEN}, headers={"X-Admin": "1"})
    assert "Secure" not in plain.headers["set-cookie"]
    https = httpx.post(admin.base + "/api/login", json={"token": TOKEN},
                       headers={"X-Admin": "1", "X-Forwarded-Proto": "https"})
    assert "Secure" in https.headers["set-cookie"]


async def test_read_page_rebinding_to_localhost_is_refused(mocker):
    """The check sees a public address, the connection would get 127.0.0.1: refused, nothing fetched."""
    import asyncio
    import socket

    from llmbot import webtools

    answers = iter(["93.184.215.14", "127.0.0.1", "127.0.0.1"])
    loop = asyncio.get_running_loop()

    async def rebinding(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), port or 0))]

    mocker.patch.object(loop, "getaddrinfo", side_effect=rebinding)
    connect = mocker.patch("httpcore._backends.auto.AutoBackend.connect_tcp")
    out = await webtools.read_page("http://rebind.example:8086/v1/contacts/x")
    assert out.startswith("Refused") and not connect.called
