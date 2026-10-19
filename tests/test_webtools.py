"""The free tools: nothing here touches the network."""
import json
from zoneinfo import ZoneInfo

import httpx
import pytest

from llmbot import webtools
from llmbot.agent import Agent, tools_for
from llmbot.config import Settings

TZ = ZoneInfo("Europe/Paris")


def settings(**overrides) -> Settings:
    base = {"signal_service": "127.0.0.1:8080", "phone_number": "+33700000000", "openrouter_api_key": ""}
    return Settings(**{**base, **overrides})


# ── offering the tools ──────────────────────────────────────────────────────

def test_web_tools_are_offered_by_default():
    names = [t["function"]["name"] for t in tools_for(settings())]
    assert names == ["get_history", "search_messages", "remind", "list_reminders", "generate_image", "get_spend",
                     "web_search", "read_page", "get_weather", "get_time", "calculate", "pick_random", "make_meme"]


def test_web_tools_can_be_switched_off():
    names = [t["function"]["name"] for t in tools_for(settings(web_tools=False))]
    assert "web_search" not in names and "calculate" not in names


def test_web_tools_env_switch():
    base = {"SIGNAL_SERVICE": "s", "PHONE_NUMBER": "p", "OPENROUTER_API_KEY": "k"}
    assert Settings.from_env(base).web_tools
    assert not Settings.from_env({**base, "WEB_TOOLS": "false"}).web_tools


async def test_agent_routes_a_web_tool():
    agent = Agent(settings(), store=None, llm=None, registry=None, tz=TZ)
    out = await agent._run_tool("calculate", {"expression": "2+2"}, req=None, images={}, result=None)
    assert out == "4"


async def test_agent_ignores_web_tools_when_off():
    agent = Agent(settings(web_tools=False), store=None, llm=None, registry=None, tz=TZ)
    out = await agent._run_tool("calculate", {"expression": "2+2"}, req=None, images={}, result=None)
    assert out.startswith("Unknown tool")


async def test_run_returns_none_for_a_foreign_tool():
    assert await webtools.run("get_spend", {}, TZ) is None


# ── calculator ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("expr,expected", [
    ("2+2", "4"),
    ("(84,5 + 12) / 3", "32.1666666667"),   # French decimal comma
    ("min(4,2)*2^3", "16"),                  # a comma between arguments stays a comma
    ("sqrt(16) + 10 % 3", "5"),
    ("100 * 1.2", "120"),
    ("-3 ** 2", "-9"),
])
def test_calculate(expr, expected):
    assert webtools.calculate(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('id')", "open('/etc/passwd')", "().__class__", "x + 1", "'a' * 3", "True + 1",
    "lambda: 1", "9**9**9", "10**100000",
])
def test_calculate_refuses_anything_but_arithmetic(expr):
    assert webtools.calculate(expr).startswith("Cannot compute")


def test_calculate_division_by_zero():
    assert webtools.calculate("1/0") == "Division by zero."


# ── time ────────────────────────────────────────────────────────────────────

def test_get_time_is_case_insensitive():
    assert webtools.get_time("asia/tokyo").endswith("(Asia/Tokyo)")


def test_get_time_suggests_the_iana_name():
    out = webtools.get_time("Tokyo")
    assert out.startswith("Unknown timezone") and "Asia/Tokyo" in out


# ── reading a page ──────────────────────────────────────────────────────────

def test_html_to_text_drops_scripts_menus_and_keeps_the_article():
    html = ("<html><head><title>T</title><script>evil()</script></head><body><nav>Menu Menu</nav>"
            "<main><h1>Hello</h1><p>" + "Body text. " * 80 + "</p></main><footer>Legal</footer></body></html>")
    out = webtools.html_to_text(html)
    assert out.startswith("T") and "Hello" in out and "Body text." in out
    assert "evil" not in out and "Menu" not in out and "Legal" not in out


@pytest.mark.parametrize("url", [
    "http://localhost:8089/v1/models", "http://127.0.0.1/", "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/", "http://192.168.1.1/", "http://[::1]/",
])
async def test_read_page_refuses_private_addresses(url):
    assert (await webtools.read_page(url)).startswith("Refused")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "example.com", ""])
async def test_read_page_refuses_other_schemes(url):
    assert (await webtools.read_page(url)).startswith("Only full http(s)")


async def test_read_page_does_not_follow_a_redirect_to_a_private_address(monkeypatch):
    async def public(host):
        return host == "example.com"
    monkeypatch.setattr(webtools, "_public_host", public)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        out = await webtools.read_page("https://example.com/x", client)
    assert out.startswith("Refused")


async def test_read_page_returns_the_text_truncated(monkeypatch):
    async def public(host):
        return True
    monkeypatch.setattr(webtools, "_public_host", public)
    page = "<html><body><p>" + "word " * 5000 + "</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=page, headers={"content-type": "text/html; charset=utf-8"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        out = await webtools.read_page("https://example.com/", client)
    assert out.startswith("word") and out.endswith("[…]") and len(out) <= webtools.PAGE_CHARS + 5


async def test_read_page_refuses_binary_content(monkeypatch):
    async def public(host):
        return True
    monkeypatch.setattr(webtools, "_public_host", public)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        out = await webtools.read_page("https://example.com/a.pdf", client)
    assert "Cannot read" in out


# ── search ──────────────────────────────────────────────────────────────────

async def test_web_search_formats_results(monkeypatch):
    monkeypatch.setattr(webtools, "_search_sync", lambda q: [
        {"title": "Ligue 1", "href": "https://example.com/l1", "body": "Monaco leads."}])
    out = await webtools.web_search("ligue 1")
    assert "Ligue 1" in out and "https://example.com/l1" in out and "Monaco leads." in out


async def test_web_search_failure_tells_the_model_not_to_guess(monkeypatch):
    def boom(q):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(webtools, "_search_sync", boom)
    assert "do not guess" in await webtools.web_search("anything")


async def test_web_search_needs_a_query():
    assert await webtools.web_search("  ") == "A query is required."


# ── weather ─────────────────────────────────────────────────────────────────

GEO = {"results": [{"name": "Lyon", "admin1": "Auvergne-Rhône-Alpes", "country": "France",
                    "latitude": 45.75, "longitude": 4.85}]}
FORECAST = {"current": {"temperature_2m": 20.0, "apparent_temperature": 21.0, "weather_code": 3,
                        "wind_speed_10m": 5.0, "precipitation": 0.0},
            "daily": {"time": ["2026-10-16", "2026-10-17"], "weather_code": [0, 61],
                      "temperature_2m_max": [24.0, 18.0], "temperature_2m_min": [12.0, 13.0],
                      "precipitation_probability_max": [0, 40], "precipitation_sum": [0.0, 1.2]}}


async def test_get_weather():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=GEO if "geocoding" in request.url.host else FORECAST)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        out = await webtools.get_weather("Lyon, France", client)
    assert out.startswith("Lyon, Auvergne-Rhône-Alpes, France. Now: 20.0°C")
    assert "overcast" in out and "light rain" in out and "40%" in out
    assert seen[0].params["name"] == "Lyon"  # the country is not part of the geocoder query


async def test_get_weather_unknown_city():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert (await webtools.get_weather("zzzz", client)).startswith("No place found")


async def test_get_weather_service_down():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert "unavailable" in await webtools.get_weather("Lyon", client)


# ── GIF ─────────────────────────────────────────────────────────────────────

GIF_BYTES = b"GIF89a" + b"\x00" * 50
SEARCH = {"result": True, "data": {"data": [
    {"id": 1, "type": "gif", "file": {
        "md": {"gif": {"url": "https://static.klipy.com/a/md.gif", "size": 1_700_000}, "mp4": {"url": "https://x/a.mp4"}},
        "sm": {"gif": {"url": "https://static.klipy.com/a/sm.gif", "size": 600_000}}}},
    {"id": 2, "type": "ad", "content": "<div>ad</div>"},
    {"id": 3, "file": {"hd": {"webp": {"url": "https://x/b.webp"}}}},
], "current_page": 1, "has_next": False}}
SECRET = "sk-secret-key"


def gif_transport(seen=None, search=SEARCH, gif=GIF_BYTES):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request.url)
        if request.url.host == "api.klipy.com":
            return httpx.Response(200, json=search)
        return httpx.Response(200, content=gif, headers={"content-type": "image/gif"})
    return httpx.MockTransport(handler)


def test_gif_tool_is_offered_only_with_a_key():
    assert "send_gif" not in [t["function"]["name"] for t in tools_for(settings())]
    assert "send_gif" in [t["function"]["name"] for t in tools_for(settings(klipy_api_key="k"))]


def test_gif_key_env():
    base = {"SIGNAL_SERVICE": "s", "PHONE_NUMBER": "p", "OPENROUTER_API_KEY": "k"}
    assert not Settings.from_env(base).gifs_enabled
    assert Settings.from_env({**base, "KLIPY_API_KEY": " abc "}).klipy_api_key == "abc"


def test_gif_url_prefers_medium_and_skips_ads_and_non_gif_results():
    items = SEARCH["data"]["data"]
    assert webtools.gif_url(items[0]) == "https://static.klipy.com/a/md.gif"
    assert webtools.gif_url(items[1]) is None   # an ad has no files
    assert webtools.gif_url(items[2]) is None   # webp only: Signal needs a real GIF


async def test_find_gif_downloads_the_gif():
    seen = []
    async with httpx.AsyncClient(transport=gif_transport(seen)) as client:
        data, message = await webtools.find_gif("facepalm", SECRET, client)
    assert data == GIF_BYTES and "will be sent" in message
    search = seen[0]
    assert search.path == f"/api/v1/{SECRET}/gifs/search"
    assert search.params["q"] == "facepalm" and search.params["customer_id"]


async def test_find_gif_with_no_usable_result():
    empty = {"result": True, "data": {"data": [SEARCH["data"]["data"][1]]}}
    async with httpx.AsyncClient(transport=gif_transport(search=empty)) as client:
        data, message = await webtools.find_gif("zzz", SECRET, client)
    assert data is None and message.startswith("No GIF found")


async def test_find_gif_refuses_a_file_that_is_not_a_gif():
    async with httpx.AsyncClient(transport=gif_transport(gif=b"<html>nope</html>")) as client:
        data, message = await webtools.find_gif("cat", SECRET, client)
    assert data is None and "not a GIF" in message


async def test_find_gif_refuses_a_huge_file(monkeypatch):
    monkeypatch.setattr(webtools, "GIF_MAX_BYTES", 10)
    async with httpx.AsyncClient(transport=gif_transport()) as client:
        data, message = await webtools.find_gif("cat", SECRET, client)
    assert data is None and "too big" in message


async def test_find_gif_failure_never_leaks_the_key(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with caplog.at_level("DEBUG"):
            data, message = await webtools.find_gif("cat", SECRET, client)
    assert data is None and "unavailable" in message
    assert SECRET not in message and SECRET not in caplog.text


async def test_agent_attaches_one_gif_per_reply():
    from llmbot.agent import AgentResult
    agent = Agent(settings(klipy_api_key=SECRET), store=None, llm=None, registry=None, tz=TZ)
    result = AgentResult(text="")

    async def fake(query, key, client=None):
        return GIF_BYTES, "GIF found"
    import llmbot.webtools as wt
    original, wt.find_gif = wt.find_gif, fake
    try:
        first = await agent._run_tool("send_gif", {"query": "yes"}, req=None, images={}, result=result)
        second = await agent._run_tool("send_gif", {"query": "no"}, req=None, images={}, result=result)
    finally:
        wt.find_gif = original
    assert first == "GIF found" and result.images == [(GIF_BYTES, "image/gif")]
    assert second.startswith("Refused") and len(result.images) == 1


async def test_agent_refuses_a_gif_without_a_key():
    agent = Agent(settings(), store=None, llm=None, registry=None, tz=TZ)
    out = await agent._run_tool("send_gif", {"query": "yes"}, req=None, images={}, result=None)
    assert out == "GIFs are off on this bot."


def test_gif_url_skips_a_heavy_medium_size_for_a_lighter_one():
    item = {"file": {"md": {"gif": {"url": "https://s/md.gif", "size": 9_000_000}},
                     "sm": {"gif": {"url": "https://s/sm.gif", "size": 2_000_000}}}}
    assert webtools.gif_url(item) == "https://s/sm.gif"


def test_gif_url_takes_the_smallest_when_all_are_heavy():
    item = {"file": {"md": {"gif": {"url": "https://s/md.gif", "size": 9_000_000}},
                     "sm": {"gif": {"url": "https://s/sm.gif", "size": 4_000_000}},
                     "xs": {"gif": {"url": "https://s/xs.gif", "size": 3_500_000}}}}
    assert webtools.gif_url(item) == "https://s/xs.gif"


# ── found by the independent review: nothing a user types may freeze the bot ─

@pytest.mark.parametrize("expr", [
    "((10**1000)**1000)**30", "(10**1000)**1000", "9**999 * 9**999 * 9**999 * 9**999 * 9**999 * 9**999",
    "10**1000 * 10**1000 * 10**1000 * 10**1000 * 10**1000", "1" + "0" * 5000, "2**1000 ** 2",
])
def test_calculate_refuses_huge_results_quickly(expr):
    import time
    t = time.time()
    out = webtools.calculate(expr)
    assert time.time() - t < 1.0
    assert out.startswith("Cannot compute"), out


def test_calculate_still_does_big_but_reasonable_numbers():
    assert webtools.calculate("2**100") == "1267650600228229401496703205376"
    assert len(webtools.calculate("10**300")) == 301


# ── memes ───────────────────────────────────────────────────────────────────

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 64


MEME_LIST = [{"id": "drake", "name": "Drakeposting", "lines": 2, "keywords": ["hotline", "bling"]},
             {"id": "boardroom", "name": "Boardroom Meeting Suggestion", "lines": 3, "keywords": ["suggestion"]},
             {"id": "db", "name": "Distracted Boyfriend", "lines": 3, "keywords": [],
              "example": {"text": ["Socialism", "The Youth", "Capitalism"]}},
             {"id": "cmm", "name": "Change My Mind", "lines": 1, "keywords": []}]


@pytest.fixture(autouse=True)
def fresh_meme_cache(monkeypatch):
    monkeypatch.setattr(webtools, "_meme_templates", (0.0, {}))


def meme_transport(seen=None, post_status=200, templates=MEME_LIST):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/templates":
            return httpx.Response(200, json=templates) if templates is not None else httpx.Response(500)
        if seen is not None:
            seen.append(request)
        if request.method == "POST":
            if post_status != 200:
                return httpx.Response(post_status, json={"error": "nope"})
            return httpx.Response(200, json={"url": "https://api.memegen.link/images/drake/a/b.png"})
        return httpx.Response(200, content=PNG_BYTES, headers={"content-type": "image/png"})
    return httpx.MockTransport(handler)


def test_meme_tool_is_offered_by_default_and_follows_the_switches():
    assert "make_meme" in [t["function"]["name"] for t in tools_for(settings())]
    assert "make_meme" not in [t["function"]["name"] for t in tools_for(settings(disabled_tools=frozenset({"make_meme"})))]
    assert "make_meme" not in [t["function"]["name"] for t in tools_for(settings(web_tools=False))]


async def test_make_meme_posts_the_captions_as_json_and_returns_the_png():
    seen = []
    async with httpx.AsyncClient(transport=meme_transport(seen)) as client:
        png, message = await webtools.make_meme(" Drake ", ["no / ?", "yes_ 100%"], client)
    assert png == PNG_BYTES and "will be sent" in message
    assert json.loads(seen[0].content) == {"template_id": "drake", "text_lines": ["no / ?", "yes_ 100%"],
                                           "extension": "png"}


async def test_make_meme_suggests_the_closest_templates_for_a_name_or_a_theme():
    async with httpx.AsyncClient(transport=meme_transport()) as client:
        png, message = await webtools.make_meme("Boardroom meeting", ["a", "b", "c", "d"], client)
        assert png is None and "boardroom (Boardroom Meeting Suggestion, 3 lines" in message
        png, message = await webtools.make_meme("zzz", ["a"], client)
        assert png is None and "no close match" in message


async def test_make_meme_takes_a_template_named_by_its_own_name_and_its_line_roles():
    seen = []
    async with httpx.AsyncClient(transport=meme_transport(seen)) as client:
        png, message = await webtools.make_meme("Distracted Boyfriend", ["le cloud à 12€", "lui", "sa copine"], client)
        assert png == PNG_BYTES and "will be sent" in message
        assert json.loads(seen[0].content)["template_id"] == "db"
    async with httpx.AsyncClient(transport=meme_transport()) as client:
        png, message = await webtools.make_meme("db", ["one", "two"], client)
        assert png is None and "db takes exactly 3 lines" in message


async def test_make_meme_refuses_a_subject_taken_from_another_conversation():
    async with httpx.AsyncClient(transport=meme_transport()) as client:
        png, message = await webtools.make_meme(
            "drake", ["le linge de Simo", "le nez de Doge"], client,
            subject="le linge de Simo", request_text="Mop fais un meme sur les chauves")
    assert png is None and "subject must come from the request" in message


async def test_make_meme_refuses_captions_that_recycle_its_own_last_jokes():
    async with httpx.AsyncClient(transport=meme_transport()) as client:
        png, message = await webtools.make_meme(
            "drake", ["le linge de Simo", "le nez de Doge"], client,
            subject="les chauves", request_text="Mop fais un meme sur les chauves",
            own_text="Simo est innocent, son linge aussi. Le nez de Doge a tranché.")
    assert png is None and "come from your own earlier replies" in message
    async with httpx.AsyncClient(transport=meme_transport()) as client:
        png, message = await webtools.make_meme(
            "drake", ["les chauves au bureau", "et le crâne qui brille"], client,
            subject="les chauves", request_text="Mop fais un meme sur les chauves",
            own_text="Simo est innocent, son linge aussi. Le nez de Doge a tranché.")
    assert png == PNG_BYTES


async def test_make_meme_takes_the_captions_as_one_multiline_string():
    async with httpx.AsyncClient(transport=meme_transport()) as client:
        png, message = await webtools.make_meme("drake", "- les chauves\n2. ils puent", client,
                                                subject="les chauves", request_text="meme sur les chauves")
    assert png == PNG_BYTES and "will be sent" in message


async def test_make_meme_refuses_a_caption_too_long_for_the_slots():
    async with httpx.AsyncClient(transport=meme_transport()) as client:
        png, message = await webtools.make_meme("drake", ["x" * 70, "ok"], client)
    assert png is None and "too long" in message and "Shorten it" in message


async def test_make_meme_still_tries_when_the_template_list_is_unavailable():
    async with httpx.AsyncClient(transport=meme_transport(templates=None)) as client:
        png, _ = await webtools.make_meme("drake", ["a", "b"], client)
    assert png == PNG_BYTES


async def test_make_meme_reports_an_unknown_template_and_missing_captions():
    async with httpx.AsyncClient(transport=meme_transport(post_status=404, templates=None)) as client:
        png, message = await webtools.make_meme("nope", ["a"], client)
    assert png is None and "Unknown template" in message
    png, message = await webtools.make_meme("../etc", ["a"])
    assert png is None and "Unknown template" in message
    png, message = await webtools.make_meme("drake", ["", " "])
    assert png is None and "Captions" in message


async def test_make_meme_survives_the_service_being_down():
    def boom(request):
        raise httpx.ConnectError("down")
    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        png, message = await webtools.make_meme("drake", ["a", "b"], client)
    assert png is None and "unavailable" in message


async def test_agent_attaches_one_meme_per_reply(monkeypatch):
    from llmbot.agent import AgentResult
    agent = Agent(settings(), store=None, llm=None, registry=None, tz=TZ)
    result = AgentResult(text="")

    async def fake(template, lines, client=None, **kwargs):
        return PNG_BYTES, "Meme made"
    monkeypatch.setattr(webtools, "make_meme", fake)
    args = {"template": "drake", "subject": "a", "lines": ["a", "b"]}
    first = await agent._run_tool("make_meme", args, req=None, images={}, result=result)
    second = await agent._run_tool("make_meme", args, req=None, images={}, result=result)
    assert first == "Meme made" and result.images == [(PNG_BYTES, "image/png")]
    assert second.startswith("Refused") and len(result.images) == 1
