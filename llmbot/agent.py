"""Natural-language mode: the model reads the request, decides what to do and
calls tools (history, image generation, spend, model switch), then answers."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from llmbot import prompts
from llmbot.personas import Registry
from llmbot.config import Settings
from llmbot.context import format_message
from llmbot.media import image_for_model, image_for_signal
from llmbot.openrouter import OpenRouter, OpenRouterError, data_url
from llmbot.store import Store, StoredMessage

logger = logging.getLogger("llmbot.agent")

SILENCE = "[rien]"
# There is no local equivalent of OpenRouter's Image API: without an image model
# the tool is not offered at all, so the model cannot promise an image it cannot make.
IMAGE_UNAVAILABLE = "Refusé : la création d'images est désactivée sur ce bot (IMAGE_MODEL vide dans .env)."
ASPECT_RATIOS = {"1:1", "1:2", "2:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "9:21", "auto"}
MAX_ROUNDS = 5
MAX_TOOL_HISTORY = 300

GET_HISTORY_TOOL = {"type": "function", "function": {
    "name": "get_history",
    "description": "Lit plus de messages de cette discussion que les derniers fournis. "
                   "Utile pour résumer une longue période ou retrouver quelque chose d'ancien.",
    "parameters": {"type": "object", "properties": {
        "count": {"type": "integer", "description": "Nombre de messages à lire (max 300)."},
        "since_hours": {"type": "number", "description": "Ne garder que les messages des N dernières heures."},
    }},
}}
GENERATE_IMAGE_TOOL = {"type": "function", "function": {
    "name": "generate_image",
    "description": "Génère une image, ou en modifie/combine en passant des images de la discussion en référence.",
    "parameters": {"type": "object", "properties": {
        "prompt": {"type": "string", "description": "Description précise de l'image voulue (ou de la modification)."},
        "aspect_ratio": {"type": "string", "enum": sorted(ASPECT_RATIOS),
                         "description": "Format. 'auto' quand on part d'une image de référence."},
        "reference_images": {"type": "array", "items": {"type": "string"},
                             "description": "Identifiants d'images de la discussion (img1, img2…) à utiliser comme référence."},
    }, "required": ["prompt"]},
}}
GET_SPEND_TOOL = {"type": "function", "function": {
    "name": "get_spend",
    "description": "Ce que le bot a coûté (aujourd'hui, 30 jours).",
    "parameters": {"type": "object", "properties": {}},
}}


def tools_for(settings: Settings) -> list[dict]:
    """The tools the model is offered, given what this bot can actually do."""
    tools = [GET_HISTORY_TOOL]
    if settings.images_enabled:
        tools.append(GENERATE_IMAGE_TOOL)
    tools.append(GET_SPEND_TOOL)
    return tools


@dataclass
class AgentResult:
    text: str
    images: list[tuple[bytes, str]] = field(default_factory=list)
    image_costs: list[tuple[str, float]] = field(default_factory=list)  # (model, cost)
    chat_cost: float = 0.0
    chat_model: str = ""
    silent: bool = False


@dataclass
class AgentRequest:
    chat_id: str
    chat_name: str | None
    user_id: str
    thread: list[StoredMessage]      # reply chain ending with the current message
    window: list[StoredMessage]      # recent messages before the thread, for context
    quoted: StoredMessage | None     # the message replied to, if any (may be in thread)
    images_remaining: int


class Agent:
    def __init__(self, settings: Settings, store: Store, llm: OpenRouter, registry: Registry, tz: ZoneInfo) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.registry = registry
        self.tz = tz

    # ── prompt building ─────────────────────────────────────────────────────

    def _system(self, req: AgentRequest) -> str:
        s = self.settings
        now = datetime.now(self.tz)
        personas = ""
        if self.registry.personas:
            personas = "\n\nPersonnages ou styles qu'on peut te demander d'adopter (applique la consigne) :\n" + "\n".join(
                f"- {c.name} : {c.prompt.replace(chr(10), ' ')[:300]}" for c in self.registry.personas)
        where = f" dans le groupe « {req.chat_name} »" if req.chat_name else " en privé"
        can: list[str] = [
            "- résumer la discussion (une période, les N derniers messages, « depuis hier ») : les derniers messages "
            "sont fournis ci-dessous, appelle get_history s'il t'en faut plus ou de plus anciens ;",
            "- répondre à une question, expliquer, donner un avis, analyser un échange, roaster quelqu'un (vanne "
            "mordante entre potes, jamais sur le physique, l'origine ou la religion), jouer un débat pour/contre, "
            "parler avec la voix d'un personnage ;",
        ]
        if s.images_enabled:
            can.append(
                "- créer une image avec generate_image, ou modifier/combiner des images de la discussion en passant "
                "leurs identifiants (img1, img2…) en référence : une seule génération par demande sauf si on en "
                "demande plusieurs explicitement ;")
        if s.transcription_enabled:
            can.append("- les vocaux sont déjà transcrits (lignes 🎤) : si on te demande ce que dit un vocal, cite-le ;")
        else:
            can.append(
                "- les vocaux ne sont pas transcrits ici : si on te demande ce que dit un vocal, dis que la "
                "transcription est coupée et qu'on peut te réécrire le contenu en texte ;")
        if not s.images_enabled:
            can.append(
                "- la création et la modification d'images sont coupées : si on t'en demande une, dis-le simplement, "
                "sans inventer d'image ;")
        can.append("- get_spend pour le coût. Ton modèle, seul le propriétaire le change : si on te demande d'en "
                   "changer, dis que ce n'est pas toi qui décides.")
        return (
            f"Tu es {s.bot_name}, un participant{where} sur Signal. On s'adresse à toi en t'appelant par ton nom, "
            "en te mentionnant, en répondant à un de tes messages ou en privé. Tu comprends la demande en langage "
            "naturel et tu agis, sans demander de confirmation quand c'est clair.\n"
            + prompts.PERSONALITY + "\n"
            "Ce que tu sais faire :\n"
            + "\n".join(can) + "\n\n"
            "Si un message est marqué [CIBLE], c'est de lui qu'on parle (« résume ça », « roast-le », « refais cette "
            "image »). Quand on te parle directement, même juste pour te saluer ou t'interpeller "
            f"(« wsh {s.bot_name} », « salut {s.bot_name} », « {s.bot_name} ? »), tu réponds. Réponds exactement "
            + SILENCE + " seulement si on parle de toi à la troisième personne sans s'adresser à toi "
            f"(« {s.bot_name} a planté hier »).\n"
            + ("Quand tu génères une image, ton texte final est une courte légende, ou rien.\n" if s.images_enabled else "")
            + prompts.COMMON_RULES.strip()
            + personas
            + f"\n\nMaintenant : {now.strftime('%A %d/%m/%Y %H:%M')} ({s.timezone})."
            + (f" Limite d'images restante pour cette personne : {req.images_remaining}." if s.images_enabled else "")
        )

    def _line(self, m: StoredMessage, images: dict[str, dict]) -> str:
        line = format_message(m, self.tz, self._reply_name)
        ids = []
        for img in m.images():
            if Path(img["path"]).exists():
                key = f"img{len(images) + 1}"
                images[key] = img
                ids.append(key)
        return line + (f" (image : {', '.join(ids)})" if ids else "")

    def _reply_name(self, m: StoredMessage) -> str | None:
        t = self.store.get(m.chat_id, m.reply_to_ts) if m.reply_to_ts else None
        return t.author_name if t else None

    def _build_messages(self, req: AgentRequest, images: dict[str, dict]) -> list[dict]:
        system = self._system(req)
        if req.window:
            system += "\n\n[CONVERSATION — derniers messages]\n" + "\n".join(self._line(m, images) for m in req.window)
        messages: list[dict] = [{"role": "system", "content": system}]

        quoted_in_thread = req.quoted is not None and any(m.ts == req.quoted.ts for m in req.thread)
        for i, m in enumerate(req.thread):
            if m.is_bot:
                messages.append({"role": "assistant", "content": m.text})
                continue
            text = self._line(m, images)
            is_last = i == len(req.thread) - 1
            if is_last and req.quoted is not None and not quoted_in_thread:
                text = "[CIBLE]\n" + self._line(req.quoted, images) + "\n\n" + text
            content: list[dict] = [{"type": "text", "text": text}]
            for img in (m.images() + (req.quoted.images() if is_last and req.quoted and not quoted_in_thread else []))[-3:]:
                path = Path(img["path"])
                if path.exists():
                    data, mt = image_for_model(path.read_bytes(), img["content_type"])
                    content.append({"type": "image_url", "image_url": {"url": data_url(data, mt)}})
            messages.append({"role": "user", "content": content})
        return messages

    # ── tools ───────────────────────────────────────────────────────────────

    async def _run_tool(self, name: str, args: dict, req: AgentRequest, images: dict[str, dict],
                        result: AgentResult) -> str:
        if name == "get_history":
            count = max(1, min(int(args.get("count") or 100), MAX_TOOL_HISTORY))
            after = None
            if args.get("since_hours"):
                after = int((time.time() - float(args["since_hours"]) * 3600) * 1000)
            before = req.thread[-1].ts
            msgs = self.store.recent(req.chat_id, count, before_ts=before, after_ts=after)
            if not msgs:
                return "Aucun message sur cette période."
            return "\n".join(self._line(m, images) for m in msgs)

        if name == "generate_image":
            if not self.settings.images_enabled:
                return IMAGE_UNAVAILABLE
            if result.images and len(result.images) >= max(1, req.images_remaining):
                return "Refusé : limite d'images atteinte pour cette personne."
            if req.images_remaining <= 0:
                return "Refusé : limite d'images atteinte pour cette personne."
            prompt = str(args.get("prompt") or "").strip()
            if not prompt:
                return "Il faut un prompt."
            refs: list[tuple[bytes, str]] = []
            unknown = []
            for key in args.get("reference_images") or []:
                img = images.get(str(key))
                if img and Path(img["path"]).exists():
                    refs.append(image_for_model(Path(img["path"]).read_bytes(), img["content_type"]))
                else:
                    unknown.append(str(key))
            if unknown:
                return f"Images inconnues : {', '.join(unknown)}. Identifiants valides : {', '.join(images) or 'aucun'}."
            ratio = str(args.get("aspect_ratio") or ("auto" if refs else "1:1"))
            if ratio not in ASPECT_RATIOS:
                ratio = "auto" if refs else "1:1"
            try:
                out = await self.llm.generate_image(model=self.settings.image_model, prompt=prompt,
                                                    aspect_ratio=ratio, references=refs[:8] or None)
            except OpenRouterError as e:
                return f"Échec de la génération : {e}"
            result.images.append(image_for_signal(out.data, out.media_type))
            result.image_costs.append((out.model, out.cost))
            return "Image générée, elle sera envoyée avec ta réponse. Réponds maintenant avec une courte légende."

        if name == "get_spend":
            now = datetime.now(self.tz)
            day = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
            month = int((now - timedelta(days=30)).timestamp() * 1000)
            return (f"Aujourd'hui : {self.store.cost_since(day, req.chat_id):.3f} $ ici, "
                    f"{self.store.cost_since(day):.3f} $ au total. 30 jours : "
                    f"{self.store.cost_since(month, req.chat_id):.3f} $ ici, {self.store.cost_since(month):.3f} $ au total.")

        return f"Outil inconnu : {name}"

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self, req: AgentRequest, *, chat_model: str) -> AgentResult:
        images: dict[str, dict] = {}
        messages = self._build_messages(req, images)
        result = AgentResult(text="", chat_model=chat_model)
        tools = tools_for(self.settings)

        for _ in range(MAX_ROUNDS):
            reply = await self.llm.chat(model=chat_model, messages=messages, temperature=0.7,
                                        max_tokens=self.settings.max_output_tokens, tools=tools)
            result.chat_cost += reply.cost
            result.chat_model = reply.model
            if not reply.tool_calls:
                result.text = reply.text + (" […]" if reply.truncated else "")
                break
            messages.append(reply.message)
            for call in reply.tool_calls:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                logger.info("tool %s %s", fn.get("name"), json.dumps(args, ensure_ascii=False)[:200])
                output = await self._run_tool(str(fn.get("name")), args, req, images, result)
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": output})
        else:
            # ran out of rounds while still calling tools: ask for the final answer without tools
            reply = await self.llm.chat(model=chat_model, messages=messages, temperature=0.7,
                                        max_tokens=self.settings.max_output_tokens)
            result.chat_cost += reply.cost
            result.text = reply.text

        # the model sometimes copies the "[HH:MM] Name : " header of the transcript it was shown
        result.text = re.sub(rf"^\s*(\[\d{{1,2}}:\d{{2}}\]\s*)?{re.escape(self.settings.bot_name)}\s*:\s*", "",
                             result.text, flags=re.IGNORECASE) if result.text.lstrip().startswith("[") else result.text
        if result.text.strip() == SILENCE or (not result.text.strip() and not result.images):
            result.silent = not result.images
            result.text = ""
        return result
