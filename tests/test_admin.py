"""The admin page: security, secrets, live changes. The server is real (an ephemeral port), nothing goes out."""
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest

from llmbot import runtime_config as rc
from llmbot.admin import AdminService
from llmbot.agent import Agent, tools_for
from llmbot.config import LiveSettings, Settings
from llmbot.openrouter import OpenRouter
from llmbot.store import Store

TOKEN = "correct horse battery"


def make_env(**overrides) -> Settings:
    base = dict(signal_service="127.0.0.1:1", phone_number="+33612345678", openrouter_api_key="sk-or-secret-0123456789",
                local_base_url="http://127.0.0.1:8089/v1", chat_model="m", bot_name="Sig", bot_aliases=["sigsig"],
                admin_token=TOKEN, admin_host="127.0.0.1", admin_port=0)
    return Settings(**{**base, **overrides})


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    def reconfigure(self, api_key, *, base_url, local) -> None:
        self.calls.append((api_key, base_url, local))


class FakeHandler:
    refreshed = 0

    def refresh(self) -> None:
        self.refreshed += 1


@pytest.fixture
def admin():
    env = make_env()
    live = LiveSettings(env)
    store = Store()
    llm, handler = FakeLLM(), FakeHandler()
    svc = AdminService(live, env, store, llm, handler)
    server = svc.start()
    svc.base = f"http://127.0.0.1:{server.server_address[1]}"
    svc.llm_, svc.handler_, svc.live_, svc.store_, svc.env_ = llm, handler, live, store, env
    yield svc
    svc.stop()


def client(svc: AdminService, login: bool = True) -> httpx.Client:
    c = httpx.Client(base_url=svc.base, headers={"X-Admin": "1"})
    if login:
        assert c.post("/api/login", json={"token": TOKEN}).status_code == 200
    return c


# ── security ────────────────────────────────────────────────────────────────

def test_page_is_public_but_the_api_is_not(admin):
    c = client(admin, login=False)
    assert c.get("/").status_code == 200 and "Réglages du bot" in c.get("/").text
    assert c.get("/api/state").status_code == 401
    assert c.put("/api/config", json={}).status_code == 401
    assert c.post("/api/restart").status_code == 401


def test_wrong_password_is_refused_and_five_tries_lock_the_address(admin):
    c = client(admin, login=False)
    for _ in range(4):
        assert c.post("/api/login", json={"token": "nope"}).status_code == 401
    assert c.post("/api/login", json={"token": "nope"}).status_code == 401  # 5th: the lock starts after it
    locked = c.post("/api/login", json={"token": TOKEN})  # even the right one waits
    assert locked.status_code == 429 and "Réessaie" in locked.json()["error"]


def test_session_cookie_is_httponly_and_samesite_strict(admin):
    r = httpx.post(admin.base + "/api/login", json={"token": TOKEN}, headers={"X-Admin": "1"})
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie


def test_changes_need_the_custom_header(admin):
    c = client(admin)
    r = httpx.put(admin.base + "/api/config", json={"changes": {"bot_name": "X"}}, cookies=c.cookies)
    assert r.status_code == 403
    assert admin.live_.bot_name == "Sig"


def test_logout_ends_the_session(admin):
    c = client(admin)
    assert c.get("/api/state").status_code == 200
    c.post("/api/logout")
    assert httpx.get(admin.base + "/api/state", cookies={"signalbot_admin": "gone"}).status_code == 401


def test_responses_carry_security_headers(admin):
    r = httpx.get(admin.base + "/")
    assert r.headers["x-frame-options"] == "DENY" and "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"


# ── secrets ─────────────────────────────────────────────────────────────────

def test_state_never_contains_a_secret(admin):
    admin.live_.update(klipy_api_key="kl-secret-key-abcdefgh")
    body = client(admin).get("/api/state").text
    assert "sk-or-secret" not in body and "kl-secret" not in body and TOKEN not in body
    fields = client(admin).get("/api/state").json()["fields"]
    assert fields["openrouter_api_key"]["is_set"] and fields["openrouter_api_key"]["hint"] == "…6789"
    assert fields["openrouter_api_key"]["value"] is None


def test_the_admin_token_cannot_be_changed_from_the_page(admin):
    r = client(admin).put("/api/config", json={"changes": {"admin_token": "x"}})
    assert r.status_code == 400 and "inconnu" in r.json()["error"]


def test_an_empty_secret_box_keeps_the_key_and_clearing_is_explicit(admin):
    c = client(admin)
    c.put("/api/config", json={"changes": {"openrouter_api_key": ""}})
    assert admin.live_.openrouter_api_key.startswith("sk-or-secret")
    r = c.put("/api/config", json={"clear_secrets": ["openrouter_api_key"]})
    assert r.status_code == 200 and admin.live_.openrouter_api_key == ""


def test_saving_does_not_log_values(admin, caplog):
    with caplog.at_level("INFO", logger="llmbot.admin"):
        client(admin).put("/api/config", json={"changes": {"klipy_api_key": "kl-super-secret-value"}})
    assert "kl-super-secret" not in caplog.text and "klipy_api_key" in caplog.text


# ── changing settings ───────────────────────────────────────────────────────

def test_a_change_applies_live_and_is_saved(admin):
    c = client(admin)
    r = c.put("/api/config", json={"changes": {"images_per_day": 3, "context_messages": 12}})
    assert r.status_code == 200 and sorted(r.json()["changed"]) == ["context_messages", "images_per_day"]
    assert admin.live_.images_per_day == 3 and admin.live_.context_messages == 12
    # survives a restart: the saved overrides are applied over the .env
    rebuilt = rc.apply(admin.env_, rc.load(admin.store_))
    assert rebuilt.images_per_day == 3
    assert r.json()["state"]["fields"]["images_per_day"]["source"] == "page"


def test_a_value_equal_to_the_env_is_not_an_override(admin):
    c = client(admin)
    c.put("/api/config", json={"changes": {"images_per_day": 3}})
    c.put("/api/config", json={"changes": {"images_per_day": admin.env_.images_per_day}})
    assert rc.load(admin.store_) == {}


def test_reset_goes_back_to_the_env(admin):
    c = client(admin)
    c.put("/api/config", json={"changes": {"bot_name": "Zed"}})
    assert admin.live_.bot_name == "Zed"
    c.put("/api/config", json={"reset": ["bot_name"]})
    assert admin.live_.bot_name == "Sig"


@pytest.mark.parametrize("changes,message", [
    ({"context_messages": 9999}, "entre 1 et 200"),
    ({"owner_number": "0612"}, "Numéro invalide"),
    ({"allowed_numbers": ["+33612345678", "bad"]}, "Numéro invalide : bad"),
    ({"timezone": "Mars/Olympus"}, "Fuseau inconnu"),
    ({"local_base_url": "ftp://x"}, "http://"),
    ({"bot_name": "  "}, "ne peut pas être vide"),
    ({"disabled_tools": ["rm_rf"]}, "Outil inconnu"),
    ({"describe_images": "yes"}, "oui ou non"),
])
def test_bad_values_are_refused_with_a_message_in_french(admin, changes, message):
    r = client(admin).put("/api/config", json={"changes": changes})
    assert r.status_code == 400 and message in r.json()["error"]
    assert rc.load(admin.store_) == {}   # nothing half saved


def test_the_bot_cannot_be_left_without_any_provider(admin):
    r = client(admin).put("/api/config", json={"changes": {"local_base_url": ""}, "clear_secrets": ["openrouter_api_key"]})
    assert r.status_code == 400 and "au moins un fournisseur" in r.json()["error"]
    assert admin.live_.local_base_url and admin.live_.openrouter_api_key  # unchanged


def test_switching_provider_rebuilds_the_client(admin):
    r = client(admin).put("/api/config", json={"changes": {"local_base_url": ""}})
    assert r.status_code == 200
    assert admin.llm_.calls == [("sk-or-secret-0123456789", "https://openrouter.ai/api/v1", False)]
    assert not admin.live_.local_chat


def test_a_new_name_refreshes_the_handler(admin):
    client(admin).put("/api/config", json={"changes": {"bot_aliases": ["a", "b"]}})
    assert admin.handler_.refreshed == 1 and admin.live_.bot_aliases == ["a", "b"]


def test_a_change_that_needs_a_restart_is_reported_until_it_is_undone(admin):
    c = client(admin)
    assert c.get("/api/state").json()["restart_pending"] == []
    r = c.put("/api/config", json={"changes": {"allowed_groups": ["Les gars"]}})
    assert r.json()["state"]["restart_pending"] == ["allowed_groups"]
    assert r.json()["state"]["fields"]["allowed_groups"]["live"] is False
    r = c.put("/api/config", json={"reset": ["allowed_groups"]})
    assert r.json()["state"]["restart_pending"] == []


def test_warns_when_images_are_wanted_without_an_openrouter_key(admin):
    r = client(admin).put("/api/config", json={"clear_secrets": ["openrouter_api_key"]})
    assert r.status_code == 200 and any("images" in w for w in r.json()["warnings"])


# ── tools ───────────────────────────────────────────────────────────────────

def test_switching_a_tool_off_removes_it_from_what_the_model_is_offered(admin):
    names = lambda: [t["function"]["name"] for t in tools_for(admin.live_)]
    assert "calculate" in names() and "get_weather" in names()
    client(admin).put("/api/config", json={"changes": {"disabled_tools": ["calculate", "generate_image"]}})
    assert "calculate" not in names() and "generate_image" not in names() and "get_weather" in names()


async def test_a_switched_off_tool_refuses_even_if_the_model_calls_it(admin):
    client(admin).put("/api/config", json={"changes": {"disabled_tools": ["calculate"]}})
    agent = Agent(admin.live_, store=None, llm=None, registry=None, tz=ZoneInfo("Europe/Paris"))
    assert await agent._run_tool("calculate", {"expression": "1+1"}, req=None, images={}, result=None) == "This tool is switched off."


def test_the_prompt_stops_mentioning_a_switched_off_tool(admin):
    agent = Agent(admin.live_, store=None, llm=None, registry=SimpleNamespace(personas=[]), tz=ZoneInfo("Europe/Paris"))
    req = SimpleNamespace(chat_name="g", context_start=None, images_remaining=3)
    assert "calculate for any arithmetic" in agent._system(req)
    client(admin).put("/api/config", json={"changes": {"disabled_tools": ["calculate", "get_spend"]}})
    prompt = agent._system(req)
    assert "calculate for any arithmetic" not in prompt and "get_spend" not in prompt and "web_search" in prompt


def test_state_explains_why_a_tool_does_not_work(admin):
    tools = {t["name"]: t for t in client(admin).get("/api/state").json()["tools"]}
    assert tools["send_gif"]["reason"] == "Il manque la clé Klipy." and not tools["send_gif"]["working"]
    assert tools["calculate"]["working"] and tools["get_history"]["locked"]
    client(admin).put("/api/config", json={"changes": {"web_tools": False}})
    tools = {t["name"]: t for t in client(admin).get("/api/state").json()["tools"]}
    assert "interrupteur général" in tools["calculate"]["reason"]


def test_a_tool_can_be_tried_from_the_page(admin):
    r = client(admin).post("/api/test/tool", json={"tool": "calculate", "input": "214,60 / 7"}).json()
    assert r["ok"] and r["text"].startswith("30.65")
    r = client(admin).post("/api/test/tool", json={"tool": "read_page", "input": "http://localhost:1/"}).json()
    assert not r["ok"] and r["text"].startswith("Refused")
    assert client(admin).post("/api/test/tool", json={"tool": "generate_image", "input": "x"}).json()["ok"] is False


# ── status ──────────────────────────────────────────────────────────────────

def test_status_reflects_the_real_database(admin):
    from llmbot.store import KIND_TEXT, StoredMessage
    for i in range(3):
        admin.store_.add(StoredMessage(chat_id="g", ts=1000 + i, author_id="u", author_name="K", kind=KIND_TEXT,
                                       text="hi", reply_to_ts=None, attachments=[]))
    st = client(admin).get("/api/state").json()["status"]
    assert st["messages"] == 3 and st["chats"] == 1 and st["provider"] == "local" and st["bot_name"] == "Sig"


def test_signal_status_when_signal_cli_is_down(admin):
    assert client(admin).get("/api/signal").json()["reachable"] is False


# ── pieces ──────────────────────────────────────────────────────────────────

def test_live_settings_replace_the_snapshot_as_a_whole():
    live = LiveSettings(make_env())
    before = live.snapshot
    live.update(bot_name="Zed")
    assert live.bot_name == "Zed" and before.bot_name == "Sig"   # a request in flight keeps a consistent view
    with pytest.raises(AttributeError):
        live.bot_name = "nope"


def test_openrouter_reconfigure_switches_endpoint_and_closes_the_old_clients_later():
    llm = OpenRouter("key", base_url="http://127.0.0.1:8089/v1", local=True)
    old = llm._client
    llm.reconfigure("key", base_url="https://openrouter.ai/api/v1", local=False)
    assert not llm.local and llm._openrouter is None and llm._client is not old
    assert old in llm._retired and not old.is_closed   # a request in flight still works


async def test_closing_the_client_closes_the_retired_ones_too():
    llm = OpenRouter("key", base_url="http://127.0.0.1:8089/v1", local=True)
    old = llm._client
    llm.reconfigure("key")
    await llm.close()
    assert old.is_closed and llm._client.is_closed


def test_the_page_file_ships_with_the_package():
    from llmbot.admin import PAGE
    html = PAGE.read_text()
    assert "<title>Réglages du bot</title>" in html and "sk-or" not in html


def test_the_real_handler_picks_up_a_new_name_and_timezone():
    from llmbot.handler import LLMHandler
    from llmbot.personas import Registry
    live = LiveSettings(make_env())
    handler = LLMHandler(live, Store(), OpenRouter("k", local=False), Registry([]))
    assert handler._name_re.search("yo sigsig") and not handler._name_re.search("yo kiki")
    live.update(bot_aliases=["kiki"], timezone="Asia/Tokyo")
    handler.refresh()
    assert handler._name_re.search("yo kiki") and not handler._name_re.search("yo sigsig")
    assert str(handler.tz) == "Asia/Tokyo" and handler.agent.tz is handler.tz


# ── found by the independent review ─────────────────────────────────────────

def test_a_restart_field_changes_the_saved_value_but_not_the_running_bot(admin):
    c = client(admin)
    r = c.put("/api/config", json={"changes": {"phone_number": "+33699999999"}})
    assert r.status_code == 200 and r.json()["changed"] == ["phone_number"]
    assert admin.live_.phone_number == "+33612345678"          # the running bot keeps answering as the old account
    state = r.json()["state"]
    assert state["fields"]["phone_number"]["value"] == "+33699999999" and state["restart_pending"] == ["phone_number"]
    assert state["status"]["phone_number"] == "+33612345678"   # what the overview shows is what runs
    assert rc.apply(admin.env_, rc.load(admin.store_)).phone_number == "+33699999999"   # the next start takes it
    r = c.put("/api/config", json={"reset": ["phone_number"]})
    assert r.json()["state"]["restart_pending"] == []


def test_a_live_field_and_a_restart_field_saved_together(admin):
    r = client(admin).put("/api/config", json={"changes": {"images_per_day": 2, "allowed_numbers": ["+33611111111"]}})
    assert sorted(r.json()["changed"]) == ["allowed_numbers", "images_per_day"]
    assert admin.live_.images_per_day == 2 and admin.live_.allowed_numbers == []


@pytest.mark.parametrize("length", ["-1", "abc", "9999999"])
def test_a_bad_content_length_is_refused_at_once(admin, length):
    import socket
    host, port = admin.base.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=3) as s:
        s.sendall(f"POST /api/login HTTP/1.1\r\nHost: x\r\nX-Admin: 1\r\nContent-Length: {length}\r\n\r\n".encode())
        assert b"400" in s.recv(200).split(b"\r\n")[0]


def test_a_refused_request_does_not_leave_its_body_to_be_read_as_the_next_request(admin):
    import http.client
    host, port = admin.base.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=3)
    conn.request("POST", "/api/config", body='{"changes": {}}')        # no X-Admin header: refused
    r = conn.getresponse()
    assert r.status == 403
    r.read()
    conn.request("GET", "/healthz")                                      # same connection
    r2 = conn.getresponse()
    assert r2.status == 200 and r2.read() == b'{"ok": true}'
    conn.close()


def test_a_provider_that_answers_json_that_is_not_an_object_is_not_a_crash(admin, monkeypatch):
    monkeypatch.setattr("llmbot.admin.httpx.get", lambda *a, **k: httpx.Response(200, json=["not", "an", "object"],
                                                                                   request=httpx.Request("GET", "http://x")))
    r = client(admin).post("/api/test/provider", json={"base_url": "http://127.0.0.1:9/v1"}).json()
    assert r["ok"] is True and r["models"] == []


def test_the_overview_hides_credentials_in_the_local_address(admin):
    client(admin).put("/api/config", json={"changes": {"local_base_url": "http://user:hunter2@10.0.0.5:8089/v1"}})
    st = client(admin).get("/api/state").json()["status"]
    assert "hunter2" not in st["endpoint"] and st["endpoint"] == "http://10.0.0.5:8089/v1"


def test_tool_rules_are_in_the_prompt_only_for_tools_that_are_on(admin):
    agent = Agent(admin.live_, store=None, llm=None, registry=SimpleNamespace(personas=[]), tz=ZoneInfo("Europe/Paris"))
    req = SimpleNamespace(chat_name="g", context_start=None, images_remaining=3)
    prompt = agent._system(req)
    assert "Tool rules, always followed" in prompt
    assert "call web_search FIRST" in prompt and "call calculate" in prompt and "call read_page" in prompt
    assert "call get_history; to find a past message by topic: call search_messages" in prompt
    client(admin).put("/api/config", json={"changes": {"disabled_tools": ["web_search", "calculate", "search_messages"]}})
    prompt = agent._system(req)
    assert "web_search FIRST" not in prompt and "call calculate" not in prompt and "search_messages" not in prompt
    assert "call get_history" in prompt and "call read_page" in prompt
    client(admin).put("/api/config", json={"changes": {"web_tools": False, "disabled_tools": []}})
    assert "call read_page" not in agent._system(req) and "call get_history" in agent._system(req)
