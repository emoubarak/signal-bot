"""Free tools that need no account or key: web search, reading a page, weather,
time in another timezone, arithmetic. Each returns plain text for the model."""
from __future__ import annotations

import ast
import asyncio
import ipaddress
import logging
import math
import operator
import random
import re
import socket
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo, available_timezones

import httpx

logger = logging.getLogger("llmbot.webtools")

SEARCH_RESULTS = 5
PAGE_CHARS = 6000
PAGE_BYTES = 600_000
PAGE_REDIRECTS = 3
HTTP_TIMEOUT = 10.0
USER_AGENT = "Mozilla/5.0 (compatible; signal-llm-bot)"

WEB_SEARCH_TOOL = {"type": "function", "function": {
    "name": "web_search",
    "description": "Searches the web. Use it for recent events, prices, scores, people, anything you do not "
                   "know for sure. Returns titles, links and short extracts.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Search query, as you would type it in a search engine."},
    }, "required": ["query"]},
}}
READ_PAGE_TOOL = {"type": "function", "function": {
    "name": "read_page",
    "description": "Reads the text of a web page, e.g. a link posted in the conversation or a search result.",
    "parameters": {"type": "object", "properties": {
        "url": {"type": "string", "description": "Full http(s) address of the page."},
    }, "required": ["url"]},
}}
GET_WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather and the next three days for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name, optionally with country (Lyon, France)."},
    }, "required": ["city"]},
}}
GET_TIME_TOOL = {"type": "function", "function": {
    "name": "get_time",
    "description": "Current date and time in a given timezone (the group's own time is already in your context).",
    "parameters": {"type": "object", "properties": {
        "timezone": {"type": "string", "description": "IANA name, e.g. Asia/Tokyo or America/New_York."},
    }, "required": ["timezone"]},
}}
CALCULATE_TOOL = {"type": "function", "function": {
    "name": "calculate",
    "description": "Evaluates an arithmetic expression exactly. Use it for any computation, splitting a bill, "
                   "percentages, conversions. Supports + - * / // % ** parentheses, sqrt, round, abs, min, max, "
                   "pi, e.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string", "description": "For example: (84.5 + 12) / 3"},
    }, "required": ["expression"]},
}}
PICK_RANDOM_TOOL = {"type": "function", "function": {
    "name": "pick_random",
    "description": "Draws at random: heads or tails, a dice roll, or one choice among several. Use it whenever the "
                   "answer must be chance (you cannot be random yourself).",
    "parameters": {"type": "object", "properties": {
        "choices": {"type": "array", "items": {"type": "string"}, "description": "Pick one of these (names, options)."},
        "dice": {"type": "integer", "description": "Roll a die with this many sides (6 for a normal die)."},
    }},
}}
SEND_GIF_TOOL = {"type": "function", "function": {
    "name": "send_gif",
    "description": "Sends an animated GIF with your reply, as a reaction. One per reply, only when a GIF says it "
                   "better than words.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "2 to 4 keywords in English describing the reaction or scene."},
    }, "required": ["query"]},
}}
TOOLS = [WEB_SEARCH_TOOL, READ_PAGE_TOOL, GET_WEATHER_TOOL, GET_TIME_TOOL, CALCULATE_TOOL, PICK_RANDOM_TOOL]
NAMES = {t["function"]["name"] for t in TOOLS}


async def run(name: str, args: dict, tz: ZoneInfo, client: httpx.AsyncClient | None = None) -> str | None:
    """Output of the tool, or None when `name` is not one of ours."""
    if name not in NAMES:
        return None
    try:
        if name == "web_search":
            return await web_search(str(args.get("query") or ""))
        if name == "read_page":
            return await read_page(str(args.get("url") or ""), client)
        if name == "get_weather":
            return await get_weather(str(args.get("city") or ""), client)
        if name == "get_time":
            return get_time(str(args.get("timezone") or ""))
        if name == "pick_random":
            return pick_random(args.get("choices"), args.get("dice"))
        return calculate(str(args.get("expression") or ""))
    except Exception as e:  # a tool must never take the whole reply down
        logger.exception("tool %s failed", name)
        return f"{name} failed: {type(e).__name__}: {e}"


# ── web search ──────────────────────────────────────────────────────────────

def _search_sync(query: str) -> list[dict]:
    from ddgs import DDGS
    return list(DDGS().text(query, max_results=SEARCH_RESULTS))


async def web_search(query: str) -> str:
    query = query.strip()
    if not query:
        return "A query is required."
    try:
        hits = await asyncio.wait_for(asyncio.to_thread(_search_sync, query), timeout=20)
    except Exception as e:
        logger.warning("search %r failed: %s", query, e)
        return f"Search failed ({type(e).__name__}). Say you could not look it up, do not guess."
    if not hits:
        return "No result."
    return "\n\n".join(f"{h.get('title', '')}\n{h.get('href', '')}\n{(h.get('body') or '').strip()}" for h in hits)


# ── reading a page ──────────────────────────────────────────────────────────

class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "template", "nav", "footer", "form", "header", "aside"}
    MAIN = {"main", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.main_parts: list[str] = []  # what sits inside <main>/<article>, preferred when present
        self.in_main = 0
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.MAIN:
            self.in_main += 1
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div", "section"}:
            self._add("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.MAIN and self.in_main:
            self.in_main -= 1

    def _add(self, text: str) -> None:
        self.parts.append(text)
        if self.in_main:
            self.main_parts.append(text)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip and data.strip():
            self._add(data.strip() + " ")


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    lines = [" ".join(line.split()) for line in "".join(p.parts).splitlines()]
    body = "\n".join(line for line in lines if line)
    main = "\n".join(l for l in (" ".join(l.split()) for l in "".join(p.main_parts).splitlines()) if l)
    if len(main) > 500:  # a real article body, not a stub <main> around a menu
        body = main
    return (f"{p.title.strip()}\n\n" if p.title.strip() else "") + body


async def _public_host(host: str) -> bool:
    """False for localhost, private networks, link-local and anything not routable on the internet."""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


async def read_page(url: str, client: httpx.AsyncClient | None = None) -> str:
    url = url.strip()
    own = client is None
    client = client or httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
    try:
        for _ in range(PAGE_REDIRECTS + 1):
            u = urlparse(url)
            if u.scheme not in ("http", "https") or not u.hostname:
                return "Only full http(s) addresses can be read."
            if not await _public_host(u.hostname):
                return "Refused: this address is not a public website."
            async with client.stream("GET", url, follow_redirects=False) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers.get("location", ""))
                    continue
                if r.status_code >= 400:
                    return f"The page answered HTTP {r.status_code}."
                ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and not (ctype.startswith("text/") or ctype in ("application/xhtml+xml", "application/json")):
                    return f"Cannot read this kind of content ({ctype})."
                raw = b""
                async for chunk in r.aiter_bytes():
                    raw += chunk
                    if len(raw) > PAGE_BYTES:
                        break
                text = raw.decode(r.encoding or "utf-8", errors="replace")
            if ctype != "text/plain" and ctype != "application/json":
                text = html_to_text(text)
            text = text.strip()
            if not text:
                return "The page has no readable text (it may need JavaScript)."
            return text[:PAGE_CHARS] + (" […]" if len(text) > PAGE_CHARS else "")
        return "Too many redirects."
    except httpx.HTTPError as e:
        return f"Could not load the page ({type(e).__name__})."
    finally:
        if own:
            await client.aclose()


# ── weather (Open-Meteo: free, no key, non-commercial use) ──────────────────

WEATHER_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "violent showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def _sky(code) -> str:
    return WEATHER_CODES.get(int(code), "unknown") if code is not None else "unknown"


async def get_weather(city: str, client: httpx.AsyncClient | None = None) -> str:
    city = city.strip()
    if not city:
        return "A city is required."
    own = client is None
    client = client or httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
    try:
        # the geocoder matches a bare place name: "Lyon, France" becomes "Lyon"
        r = await client.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": city.split(",")[0].strip(), "count": 5, "language": "en"})
        r.raise_for_status()
        places = r.json().get("results") or []
        if not places:
            return f"No place found for {city!r}."
        wanted = city.split(",")[1].strip().lower() if "," in city else ""
        place = next((p for p in places if wanted and wanted in (p.get("country", "") + " " + p.get("admin1", "")).lower()),
                     places[0])
        r = await client.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto", "forecast_days": 3,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,precipitation",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
        })
        r.raise_for_status()
        j = r.json()
    except httpx.HTTPError as e:
        return f"Weather service unavailable ({type(e).__name__})."
    finally:
        if own:
            await client.aclose()
    c, d = j["current"], j["daily"]
    where = ", ".join(x for x in (place.get("name"), place.get("admin1"), place.get("country")) if x)
    out = [f"{where}. Now: {c['temperature_2m']}°C (feels {c['apparent_temperature']}°C), {_sky(c['weather_code'])}, "
           f"wind {c['wind_speed_10m']} km/h, rain {c['precipitation']} mm."]
    for i, day in enumerate(d["time"]):
        out.append(f"{day}: {_sky(d['weather_code'][i])}, {d['temperature_2m_min'][i]} to {d['temperature_2m_max'][i]}°C, "
                   f"rain chance {d['precipitation_probability_max'][i]}% ({d['precipitation_sum'][i]} mm).")
    return "\n".join(out)


# ── time ────────────────────────────────────────────────────────────────────

def get_time(timezone: str) -> str:
    name = timezone.strip()
    zones = {z.lower(): z for z in available_timezones()}
    key = zones.get(name.lower()) or zones.get(name.lower().replace(" ", "_"))
    if not key:
        close = sorted(z for z in zones.values() if name.lower().replace(" ", "_") in z.lower())[:5]
        return f"Unknown timezone {name!r}." + (f" Did you mean: {', '.join(close)}?" if close else
                                                " Use an IANA name like Asia/Tokyo.")
    return datetime.now(ZoneInfo(key)).strftime(f"%A %d/%m/%Y %H:%M ({key})")


# ── chance ──────────────────────────────────────────────────────────────────

def pick_random(choices=None, dice=None) -> str:
    items = [str(c).strip() for c in choices if str(c).strip()][:50] if isinstance(choices, list) else []
    if items:
        return "Picked: " + random.choice(items)
    try:
        sides = int(dice) if dice not in (None, "") else 0
    except (TypeError, ValueError):
        return "`dice` must be a number of sides."
    if sides:
        if not 2 <= sides <= 1000:
            return "A die has between 2 and 1000 sides."
        return f"Rolled a d{sides}: {random.randint(1, sides)}"
    return "Flipped a coin: " + random.choice(["heads", "tails"])


# ── arithmetic ──────────────────────────────────────────────────────────────

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max}
_NAMES = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000
MAX_BITS = 14_000   # about 4200 digits: far more than a chat needs, and under Python's str() limit


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ValueError("exponent too large")
            # nested powers (10**1000)**1000 stay under the exponent limit but would freeze the bot: bound the size
            if isinstance(left, int) and isinstance(right, int) and right > 0 and abs(left) > 1 \
                    and right * abs(left).bit_length() > MAX_BITS:
                raise ValueError("result too large")
        value = _BIN[type(node.op)](left, right)
        if isinstance(value, int) and value.bit_length() > MAX_BITS:
            raise ValueError("result too large")
        return value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    text = expression.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    if not re.search(r"[a-z]\s*\(", text):  # "3,5" is a French decimal, but min(1,2) keeps its comma
        text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    if not text:
        return "An expression is required."
    try:
        value = _eval(ast.parse(text, mode="eval"))
        shown = str(value)
    except ZeroDivisionError:
        return "Division by zero."
    except (ValueError, TypeError, SyntaxError, OverflowError) as e:
        return f"Cannot compute {expression!r}: {e}"
    if isinstance(value, float):
        value = round(value, 10)
        if value == int(value) and abs(value) < 1e15:
            value = int(value)
        shown = str(value)
    return shown


# ── GIF (Klipy: free key from partner.klipy.com, 100 requests/hour until production access) ──

KLIPY_URL = "https://api.klipy.com/api/v1/{key}/gifs/search"
_KEY_IN_PATH = re.compile(r"(api\.klipy\.com/api/v1/)[^/\s?]+")


class _HideKey(logging.Filter):
    """The Klipy key is part of the URL, and httpx logs URLs at INFO: mask it whatever the log level."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(_KEY_IN_PATH.sub(r"\1***", str(a)) if "api.klipy.com" in str(a) else a
                                for a in record.args)
        return True


logging.getLogger("httpx").addFilter(_HideKey())
GIF_RATING = "pg-13"
GIF_CHOICES = 5
GIF_PREFERRED_BYTES = 3_000_000
GIF_MAX_BYTES = 8_000_000
GIF_SIZES = ("md", "sm", "hd", "xs")  # medium first: light enough for a chat, still sharp


def _file_url(node) -> str | None:
    if isinstance(node, str):
        return node if node.startswith("https://") else None
    if isinstance(node, dict):
        return _file_url(node.get("url"))
    return None


def gif_url(item: dict) -> str | None:
    """The GIF file of one search result (file.<size>.gif.url, sizes tried from medium to small): the first one
    light enough for a chat, else the smallest available."""
    files = item.get("file") or item.get("files")
    if not isinstance(files, dict):
        return None
    found: list[tuple[int, str]] = []
    for size in GIF_SIZES:
        gif = (files.get(size) or {}).get("gif") if isinstance(files.get(size), dict) else None
        if url := _file_url(gif):
            found.append((int(gif.get("size") or 0) if isinstance(gif, dict) else 0, url))
    if not found:
        return _file_url(files.get("gif"))
    for weight, url in found:
        if weight <= GIF_PREFERRED_BYTES:
            return url
    return min(found, key=lambda f: f[0] or GIF_MAX_BYTES)[1]


async def find_gif(query: str, api_key: str, client: httpx.AsyncClient | None = None) -> tuple[bytes | None, str]:
    """(gif bytes, message for the model). The key sits in the URL path: never log an httpx error."""
    query = query.strip()
    if not query:
        return None, "A query is required."
    own = client is None
    client = client or httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
    try:
        r = await client.get(KLIPY_URL.format(key=api_key), params={
            "q": query, "per_page": 8, "customer_id": "signal-bot", "rating": GIF_RATING, "locale": "us"})
        r.raise_for_status()
        items = ((r.json().get("data") or {}).get("data")) or []
        urls = [u for u in (gif_url(i) for i in items if isinstance(i, dict)) if u][:GIF_CHOICES]
        if not urls:
            return None, "No GIF found for that query. Answer with words."
        url = random.choice(urls)
        async with client.stream("GET", url) as g:
            g.raise_for_status()
            raw = b""
            async for chunk in g.aiter_bytes():
                raw += chunk
                if len(raw) > GIF_MAX_BYTES:
                    return None, "The GIF was too big. Answer with words."
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("gif search failed: %s", type(e).__name__)  # not str(e): it carries the URL, so the key
        return None, "GIF service unavailable. Answer with words."
    finally:
        if own:
            await client.aclose()
    if not raw.startswith(b"GIF8"):
        return None, "The file was not a GIF. Answer with words."
    return raw, "GIF found, it will be sent with your reply. Now reply with a few words at most, or nothing."
