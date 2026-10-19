"""The admin web page: configure the bot without touching the .env.

A small stdlib HTTP server in a thread of its own, started only when ADMIN_TOKEN is set.
It reads and writes the same LiveSettings the bot runs with, so most changes apply at once.
Nothing here ever sends a secret back to the browser: it only says whether one is set."""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import re
import secrets
import signal
import threading
import time
from dataclasses import replace
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import httpx

from llmbot import runtime_config as rc
from llmbot import webtools
from llmbot.agent import tools_for
from llmbot.config import LiveSettings, Settings
from llmbot.openrouter import BASE_URL, OpenRouter
from llmbot.store import Store

logger = logging.getLogger("llmbot.admin")

PAGE = Path(__file__).parent / "static" / "admin.html"
COOKIE = "signalbot_admin"
SESSION_SECONDS = 7 * 86400
MAX_BODY = 64_000
LOGIN_FAILS = 5
LOGIN_LOCK_SECONDS = 60
CATALOG_TTL = 1800

# label, one-line help (French: this page is read by a person), group
TOOL_INFO = {
    "generate_image": ("Créer des images", "Dessine ou retouche une image à la demande. Payant, à l'image, via OpenRouter.", "Images"),
    "send_gif": ("Envoyer un GIF", "Réagit avec un GIF animé trouvé sur Klipy. Gratuit.", "Images"),
    "make_meme": ("Faire un meme", "Répond avec un meme (Drake, Distracted Boyfriend…) légendé sur le sujet demandé, avec memegen.link. Gratuit.", "Images"),
    "web_search": ("Chercher sur internet", "Trouve une info récente (score, prix, actualité) avec DuckDuckGo. Gratuit.", "Internet"),
    "read_page": ("Lire un lien", "Lit une page web collée dans le groupe pour la résumer. Gratuit.", "Internet"),
    "get_weather": ("Météo", "Météo du jour et des trois prochains jours pour une ville, avec Open-Meteo. Gratuit.", "Internet"),
    "get_time": ("Heure ailleurs", "Donne l'heure qu'il est dans une autre ville du monde.", "Utilitaires"),
    "calculate": ("Calculatrice", "Fait les calculs exactement : addition au resto, pourcentages, conversions.", "Utilitaires"),
    "pick_random": ("Tirer au sort", "Pile ou face, dé, choix au hasard entre plusieurs options : le modèle ne sait pas faire le hasard seul.", "Utilitaires"),
    "search_messages": ("Retrouver un message", "Cherche par mots-clés ce que quelqu'un a dit plus tôt dans la conversation.", "Conversation"),
    "remind": ("Rappels", "Programme un message dans le groupe à une heure donnée (« rappelle-nous à 20 h »). Les rappels survivent à un redémarrage.", "Conversation"),
    "get_spend": ("Coût du bot", "Dit combien le bot a coûté aujourd'hui et sur 30 jours.", "Utilitaires"),
}
ALWAYS_ON = ("get_history", "Relire l'historique", "Retrouve ce qui s'est dit plus tôt dans le groupe.", "Utilitaires")


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        """A browser that closes the tab mid-answer is not an error worth a traceback on stderr."""
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            logger.debug("admin: client %s went away (%s)", client_address[0], type(exc).__name__)
        else:
            logger.exception("admin: request from %s failed", client_address[0])


class AdminService:
    def __init__(self, live: LiveSettings, base_env: Settings, store: Store, llm: OpenRouter, handler,
                 started_at: float | None = None) -> None:
        self.live = live
        self.base_env = base_env          # the .env alone, to show where each value comes from
        self.store = store
        self.llm = llm
        self.handler = handler
        self.started_at = started_at or time.time()
        self.started_with = live.snapshot  # what the running process was built with, to tell what waits for a restart
        self._lock = threading.Lock()      # one change at a time; also guards the sessions below
        self._sessions: dict[str, float] = {}
        self._fails: dict[str, tuple[int, float]] = {}
        self._catalog: tuple[float, dict] | None = None
        self.server: _Server | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> _Server:
        s = self.live.snapshot
        service = self

        class Handler(_Handler):
            admin = service

        self.server = _Server((s.admin_host, s.admin_port), Handler)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, name="admin-page", daemon=True).start()
        logger.info("admin page on http://%s:%s", s.admin_host, self.server.server_address[1])
        return self.server

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    # ── authentication ──────────────────────────────────────────────────────

    def login(self, token: str, ip: str) -> str | None:
        """A new session id when the token is right; None when it is wrong. Five wrong tries lock the address for a minute."""
        now = time.time()
        with self._lock:
            fails, until = self._fails.get(ip, (0, 0.0))
            if until > now:
                raise PermissionError(f"Trop d'essais. Réessaie dans {int(until - now) + 1} s.")
            expected = self.live.admin_token.encode()
            if expected and hmac.compare_digest(token.encode(), expected):
                self._fails.pop(ip, None)
                self._sessions = {k: v for k, v in self._sessions.items() if v > now}
                sid = secrets.token_urlsafe(32)
                self._sessions[sid] = now + SESSION_SECONDS
                return sid
            fails += 1
            self._fails[ip] = (0, now + LOGIN_LOCK_SECONDS) if fails >= LOGIN_FAILS else (fails, 0.0)
            return None

    def valid_session(self, sid: str | None) -> bool:
        with self._lock:
            return bool(sid) and self._sessions.get(sid, 0) > time.time()

    def logout(self, sid: str | None) -> None:
        with self._lock:
            self._sessions.pop(sid or "", None)

    # ── state shown on the page ─────────────────────────────────────────────

    def state(self) -> dict:
        s = self.live.snapshot
        overrides = rc.load(self.store)
        saved = rc.apply(self.base_env, overrides)  # what the next start will run with; differs from `s` only for restart fields
        fields = {}
        for key, f in rc.FIELDS.items():
            value = getattr(saved, key)
            entry = {"kind": f.kind, "live": f.live, "source": "page" if key in overrides else "env",
                     "env_value": None if key in rc.SECRETS else getattr(self.base_env, key)}
            if key in rc.SECRETS:
                entry.update(is_set=bool(value), hint=_hint(value),
                             env_set=bool(getattr(self.base_env, key)))
                entry["value"] = None
            else:
                entry["value"] = list(value) if isinstance(value, list) else value
            fields[key] = entry
        pending = [k for k, f in rc.FIELDS.items()
                   if not f.live and getattr(saved, k) != getattr(self.started_with, k)]
        return {
            "status": self._status(s),
            "fields": fields,
            "tools": self._tools(s),
            "restart_pending": pending,
            "tool_switches": rc.TOOL_SWITCHES,
        }

    def _status(self, s: Settings) -> dict:
        db = self.store._db
        day = int((time.time() - 86400) * 1000)
        month = int((time.time() - 30 * 86400) * 1000)
        with self._lock:
            n_messages = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            n_chats = db.execute("SELECT COUNT(DISTINCT chat_id) FROM messages").fetchone()[0]
            last = db.execute("SELECT MAX(ts) FROM messages").fetchone()[0]
            cost_day = self.store.cost_since(day)
            cost_month = self.store.cost_since(month)
        return {
            "bot_name": s.bot_name,
            "uptime_seconds": int(time.time() - self.started_at),
            "provider": "local" if s.local_chat else "openrouter",
            "endpoint": _hide_credentials(s.local_base_url) or "openrouter.ai",
            "chat_model": s.chat_model,
            "messages": n_messages, "chats": n_chats, "last_message_ts": last,
            "cost_24h": round(cost_day, 4), "cost_30d": round(cost_month, 4),
            "signal_service": s.signal_service, "phone_number": s.phone_number,
        }

    def _tools(self, s: Settings) -> list[dict]:
        offered = {t["function"]["name"] for t in tools_for(s)}
        out = [{"name": ALWAYS_ON[0], "label": ALWAYS_ON[1], "help": ALWAYS_ON[2], "group": ALWAYS_ON[3],
                "enabled": True, "locked": True, "reason": ""}]
        for name in rc.TOOL_SWITCHES:
            label, help_, group = TOOL_INFO[name]
            wanted = s.tool_enabled(name)
            reason = ""
            if wanted and name not in offered:
                if name == "generate_image":
                    reason = "Choisis un modèle d'image dans Fournisseur."
                elif name == "send_gif":
                    reason = "Il manque la clé Klipy."
                elif name in {t["function"]["name"] for t in webtools.TOOLS} | {"make_meme"}:
                    reason = "Les outils internet sont éteints (interrupteur général)."
            out.append({"name": name, "label": label, "help": help_, "group": group, "enabled": wanted,
                        "locked": False, "reason": reason, "working": name in offered})
        return out

    # ── changing settings ───────────────────────────────────────────────────

    def apply_changes(self, body: dict) -> dict:
        with self._lock:
            overrides = rc.load(self.store)
            for key in body.get("reset") or []:
                if key not in rc.FIELDS:
                    raise rc.ConfigError(f"Réglage inconnu : {key}")
                overrides.pop(key, None)
            for key in body.get("clear_secrets") or []:
                if key not in rc.SECRETS:
                    raise rc.ConfigError(f"Ce réglage n'est pas une clé : {key}")
                overrides[key] = ""
            for key, raw in (body.get("changes") or {}).items():
                value = rc.coerce(key, raw)
                if key == "disabled_tools":
                    value = rc.validate_disabled_tools(value)
                if key in rc.SECRETS and value == "":
                    continue  # an empty box means "keep the key", clearing is explicit
                overrides[key] = value
            for key in list(overrides):  # a value equal to the .env is not an override
                env_value = getattr(self.base_env, key)
                if overrides[key] == (list(env_value) if isinstance(env_value, list) else env_value):
                    del overrides[key]
            before_saved = rc.apply(self.base_env, rc.load(self.store))
            effective = rc.apply(self.base_env, overrides)
            warnings = rc.check(effective)
            old = self.live.snapshot
            rc.save(self.store, overrides)
            # the running bot takes only what can change under its feet; the rest waits for the next start
            running = replace(effective, **{k: getattr(self.started_with, k) for k, f in rc.FIELDS.items() if not f.live})
            self.live.replace_with(running)
            changed = rc.changed_keys(before_saved, effective)
            self._after_apply(old, running)
        logger.info("admin: changed %s", ", ".join(changed) or "nothing")  # names only, never values
        return {"changed": changed, "warnings": warnings}

    def _after_apply(self, old: Settings, new: Settings) -> None:
        if (old.local_base_url, old.openrouter_api_key) != (new.local_base_url, new.openrouter_api_key):
            self.llm.reconfigure(new.openrouter_api_key, base_url=new.local_base_url or BASE_URL, local=new.local_chat)
        if (old.bot_name, old.bot_aliases, old.timezone) != (new.bot_name, new.bot_aliases, new.timezone):
            self.handler.refresh()

    # ── tests ───────────────────────────────────────────────────────────────

    def test_provider(self, base_url: str | None, api_key: str | None) -> dict:
        """Try a provider with the values on the page (not saved yet) or, for a secret left empty, the saved ones."""
        s = self.live.snapshot
        base = s.local_base_url if base_url is None else base_url
        key = api_key or s.openrouter_api_key
        try:
            if base:
                base = rc.coerce("local_base_url", base)
                r = httpx.get(base + "/models", timeout=8)
                r.raise_for_status()
                body = r.json()
                data = body.get("data") if isinstance(body, dict) else None
                models = sorted({m.get("id") for m in (data if isinstance(data, list) else []) if isinstance(m, dict) and m.get("id")})
                return {"ok": True, "message": f"Le serveur répond ({len(models)} modèle(s) disponible(s)).", "models": models}
            if not key:
                return {"ok": False, "message": "Aucune clé OpenRouter à essayer."}
            r = httpx.get(BASE_URL + "/key", headers={"Authorization": f"Bearer {key}"}, timeout=8)
            if r.status_code in (401, 403):
                return {"ok": False, "message": "OpenRouter refuse cette clé."}
            r.raise_for_status()
            return {"ok": True, "message": "OpenRouter accepte la clé.", "models": []}
        except rc.ConfigError as e:
            return {"ok": False, "message": str(e)}
        except (httpx.HTTPError, ValueError) as e:
            return {"ok": False, "message": f"Pas de réponse ({type(e).__name__}). Vérifie l'adresse et que le serveur tourne."}

    def models_catalog(self) -> dict:
        """Model ids OpenRouter offers, for the suggestions in the model boxes (cached)."""
        now = time.time()
        if self._catalog and now - self._catalog[0] < CATALOG_TTL:
            return self._catalog[1]
        try:
            r = httpx.get(BASE_URL + "/models", timeout=10)
            r.raise_for_status()
            chat, image, audio = [], [], []
            for m in r.json().get("data", []):
                arch = m.get("architecture") or {}
                mid = m.get("id")
                if not mid:
                    continue
                if "image" in (arch.get("output_modalities") or []):
                    image.append(mid)
                if "audio" in (arch.get("input_modalities") or []):
                    audio.append(mid)
                if "text" in (arch.get("output_modalities") or ["text"]):
                    chat.append(mid)
            catalog = {"chat": sorted(chat), "image": sorted(image), "audio": sorted(audio)}
        except (httpx.HTTPError, ValueError):
            catalog = {"chat": [], "image": [], "audio": []}
        self._catalog = (now, catalog)
        return catalog

    def test_tool(self, name: str, text: str) -> dict:
        s = self.live.snapshot
        if name == "get_spend":
            db_day = int((time.time() - 86400) * 1000)
            return {"ok": True, "text": f"Dernières 24 h : {self.store.cost_since(db_day):.3f} $."}
        if name == "send_gif":
            if not s.klipy_api_key:
                return {"ok": False, "text": "Il manque la clé Klipy."}
            data, message = asyncio.run(webtools.find_gif(text or "hello", s.klipy_api_key))
            if data is None:
                return {"ok": False, "text": message}
            return {"ok": True, "text": f"GIF trouvé ({len(data) // 1024} Ko).",
                    "gif": "data:image/gif;base64," + base64.b64encode(data).decode() if len(data) < 3_000_000 else None}
        if name == "make_meme":
            data, message = asyncio.run(webtools.make_meme("drake", [x.strip() for x in text.split(",")][:2]))
            if data is None:
                return {"ok": False, "text": message}
            return {"ok": True, "text": f"Meme fait ({len(data) // 1024} Ko).",
                    "gif": "data:image/png;base64," + base64.b64encode(data).decode()}
        if name not in {t["function"]["name"] for t in webtools.TOOLS}:
            return {"ok": False, "text": "Cet outil ne s'essaie pas d'ici."}
        if name == "pick_random":
            args = {"choices": [c.strip() for c in text.split(",") if c.strip()]}
        else:
            arg = {"web_search": "query", "read_page": "url", "get_weather": "city", "get_time": "timezone",
                   "calculate": "expression"}[name]
            args = {arg: text}
        out = asyncio.run(webtools.run(name, args, None))
        failed = out.startswith(("Search failed", "Refused", "Cannot", "Unknown", "No place", "Could not", "Weather service",
                                 "Only full", "A query", "A city", "An expression", "Division", "Too many", "The page"))
        return {"ok": not failed, "text": out[:2500]}

    def signal_status(self) -> dict:
        s = self.live.snapshot
        base = f"http://{s.signal_service}"
        out: dict = {"service": s.signal_service, "phone_number": s.phone_number, "reachable": False, "accounts": []}
        try:
            out["reachable"] = httpx.get(base + "/v1/health", timeout=3).is_success
            if out["reachable"]:
                acc = httpx.get(base + "/v1/accounts", timeout=3).json()
                out["accounts"] = [a if isinstance(a, str) else str(a) for a in acc] if isinstance(acc, list) else []
                out["registered"] = s.phone_number in out["accounts"]
        except (httpx.HTTPError, ValueError):
            pass
        return out

    def signal_qr(self) -> bytes | None:
        s = self.live.snapshot
        try:
            r = httpx.get(f"http://{s.signal_service}/v1/qrcodelink", params={"device_name": s.bot_name or "signal-bot"}, timeout=10)
            return r.content if r.is_success and r.headers.get("content-type", "").startswith("image/") else None
        except httpx.HTTPError:
            return None

    def restart(self) -> None:
        """Ask the process to stop the way a deploy does (answers in progress are finished first).
        The container's restart policy brings it back."""
        threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()


def _hide_credentials(url: str) -> str:
    """http://user:pass@host/x -> http://host/x (the overview shows the address to anyone looking over a shoulder)."""
    return re.sub(r"^(https?://)[^/@\s]*@", r"\1", url)


def _hint(secret: str) -> str:
    return ("…" + secret[-4:]) if len(secret) >= 12 else ("•" * len(secret))


class _Handler(BaseHTTPRequestHandler):
    admin: AdminService
    server_version = "signal-bot-admin"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args) -> None:  # the default logger prints every request line; keep the log quiet
        pass

    # helpers
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                         "script-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data, extra: dict | None = None) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", extra)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if not 0 <= n <= MAX_BODY:
            self.close_connection = True   # the body is not read: this connection cannot be reused
            raise rc.ConfigError("Requête invalide.")
        raw = self.rfile.read(n) if n else b"{}"
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise rc.ConfigError("Requête invalide.")
        return data

    def _session(self) -> str | None:
        jar = SimpleCookie(self.headers.get("Cookie", ""))
        return jar[COOKIE].value if COOKIE in jar else None

    def _authed(self) -> bool:
        return self.admin.valid_session(self._session())

    def _same_site(self) -> bool:
        """Changes only come from the page itself: a custom header a foreign site cannot add."""
        return self.headers.get("X-Admin") == "1"

    # routes
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/healthz":
            self._json(200, {"ok": True})
            return
        if not self._authed():
            self._json(401, {"error": "Connexion requise."})
            return
        if path == "/api/state":
            self._json(200, self.admin.state())
        elif path == "/api/signal":
            self._json(200, self.admin.signal_status())
        elif path == "/api/models":
            self._json(200, self.admin.models_catalog())
        elif path == "/api/signal/qr":
            png = self.admin.signal_qr()
            if png is None:
                self._json(502, {"error": "Le QR code n'est pas disponible (signal-cli ne répond pas)."})
            else:
                self._send(200, png, "image/png")
        else:
            self._json(404, {"error": "Introuvable."})

    def do_POST(self) -> None:
        self._write("POST")

    def do_PUT(self) -> None:
        self._write("PUT")

    def _write(self, method: str) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()  # always read the body first: an unread one would be taken for the next request
            if not self._same_site():
                self._json(403, {"error": "Requête refusée."})
                return
            if path == "/api/login" and method == "POST":
                ip = self.client_address[0]
                try:
                    sid = self.admin.login(str(body.get("token") or ""), ip)
                except PermissionError as e:
                    self._json(429, {"error": str(e)})
                    return
                if sid is None:
                    self._json(401, {"error": "Mot de passe incorrect."})
                    return
                cookie = f"{COOKIE}={sid}; Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_SECONDS}"
                self._json(200, {"ok": True}, {"Set-Cookie": cookie})
                return
            if not self._authed():
                self._json(401, {"error": "Connexion requise."})
                return
            if path == "/api/logout" and method == "POST":
                self.admin.logout(self._session())
                self._json(200, {"ok": True}, {"Set-Cookie": f"{COOKIE}=; Path=/; Max-Age=0"})
            elif path == "/api/config" and method == "PUT":
                report = self.admin.apply_changes(body)
                self._json(200, {**report, "state": self.admin.state()})
            elif path == "/api/test/provider" and method == "POST":
                self._json(200, self.admin.test_provider(body.get("base_url"), body.get("api_key")))
            elif path == "/api/test/tool" and method == "POST":
                self._json(200, self.admin.test_tool(str(body.get("tool") or ""), str(body.get("input") or "")))
            elif path == "/api/restart" and method == "POST":
                self.admin.restart()
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "Introuvable."})
        except rc.ConfigError as e:
            self._json(400, {"error": str(e)})
        except json.JSONDecodeError:
            self._json(400, {"error": "Requête invalide."})
        except Exception:
            logger.exception("admin request failed")
            self._json(500, {"error": "Erreur inattendue, voir les journaux du bot."})
