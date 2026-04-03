from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple
import time as _time

from signalbot import Command, Context
import logging
import json

logger = logging.getLogger("utils.history")

# Stored message type: (name, text, timestamp_ms)
StoredMessage = Tuple[str, str, float]


def _conversation_key_from_context(c: Context) -> str:
    """
    Stable and non-colliding conversation key:
    - Group  -> "group:<group_id|internal_id>"
    - Private -> "user:<uuid|number>"
    """
    msg = c.message

    # 1) Group: depending on versions, `message.group` can be:
    #    - an object with .id
    #    - a string "internal_id" (often ends with '=')
    group_ref = getattr(msg, "group", None)
    if group_ref:
        # a) object with .id
        if hasattr(group_ref, "id"):
            gid = getattr(group_ref, "id", None)
            if gid:
                return f"group:{gid}"

        # b) string internal_id -> try to resolve "external" id via bot
        if isinstance(group_ref, str):
            gid = None
            try:
                gid = getattr(getattr(c, "bot", None), "_groups_by_internal_id", {}).get(group_ref, {}).get("id")
            except Exception:
                gid = None
            # fallback: use internal_id if no mapping
            return f"group:{gid or group_ref}"

        # c) generic fallback
        return f"group:{str(group_ref)}"

    # 2) Private: use UUID first, then number
    src = (
        getattr(msg, "source", None)
        or getattr(msg, "sourceUuid", None)
        or getattr(msg, "sourceNumber", None)
    )
    return f"user:{src or 'unknown'}"


def _extract_timestamp_from_context(c: Context) -> float:
    """
    Extracts the timestamp from the current message via raw_message or message.timestamp.
    Returns a timestamp in ms (or time.time()*1000 as fallback).
    """
    # Try from message.timestamp (Signal gives ms)
    ts = getattr(c.message, "timestamp", None)
    if ts:
        try:
            return float(ts)
        except (TypeError, ValueError):
            pass

    # Try from raw_message
    raw = getattr(c, "raw_message", None)
    if raw:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {}
        if isinstance(raw, dict):
            ts = raw.get("envelope", {}).get("timestamp")
            if ts:
                try:
                    return float(ts)
                except (TypeError, ValueError):
                    pass

    return _time.time() * 1000


def extract_quote_info(c: Context) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts quoted text AND author from raw_message.
    Returns (quote_text, quote_author) or (None, None).
    """
    raw = getattr(c, "raw_message", None)
    if not raw:
        return None, None

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None, None

    if not isinstance(raw, dict):
        return None, None

    quote = None
    # dataMessage path
    try:
        quote = (raw.get("envelope", {}).get("dataMessage", {}) or {}).get("quote", {})
    except Exception:
        pass

    # syncMessage path (fallback)
    if not quote or not (quote.get("text") or quote.get("body")):
        try:
            quote = ((raw.get("envelope", {}).get("syncMessage", {}) or {}).get("sentMessage", {}) or {}).get("quote", {})
        except Exception:
            pass

    if not quote:
        return None, None

    text = quote.get("text") or quote.get("body")
    author = quote.get("author") or quote.get("authorNumber") or quote.get("authorUuid")

    # Try to resolve author name from history
    if author and text:
        key = _conversation_key_from_context(c)
        store = HistoryStore._store.get(key, deque())
        for name, stored_text, _ in store:
            if stored_text and stored_text.strip() == text.strip():
                author = name
                break

    return text, author


class HistoryStore:
    """
    Circular RAM memory per conversation.
    Each entry is (name, text, timestamp_ms).
    """
    _store: Dict[str, Deque[StoredMessage]] = defaultdict(lambda: deque(maxlen=300))

    @classmethod
    def add(cls, c: Context, text: Optional[str], name_override: Optional[str] = None) -> None:
        if not text:
            return
        key = _conversation_key_from_context(c)
        ts = _extract_timestamp_from_context(c)

        # If name is forced (for bot), use it directly
        if name_override:
            name = name_override
        else:
            name = None

        # 1) Classic attempt via message attributes
        if not name:
            try:
                author = getattr(c.message, "author", None)
                if isinstance(author, str):
                    name = author
                elif hasattr(author, "name"):
                    name = author.name
                elif hasattr(author, "__dict__") and "name" in author.__dict__:
                    name = author.__dict__["name"]
            except Exception as e:
                logger.warning(f"[HistoryStore] Error extracting name via author: {e}")

        # 2) If still nothing, check raw_message (string JSON)
        if not name:
            try:
                raw = getattr(c, "raw_message", None)
                if isinstance(raw, str):
                    raw = json.loads(raw)
                if isinstance(raw, dict):
                    name = raw.get("envelope", {}).get("sourceName", None)
            except Exception as e:
                logger.warning(f"[HistoryStore] Error parsing raw_message for name: {e}")

        # 3) Fallback
        if not name:
            name = getattr(c.message, "source", "??")

        cls._store[key].append((name, text, ts))
        logger.info(f"[HistoryStore] Added to {key}: {name}: {text[:80]}")

    @classmethod
    def get_recent_for_context(cls, c: Context, n: int) -> List[StoredMessage]:
        key = _conversation_key_from_context(c)
        buf = cls._store.get(key, deque())
        if not buf:
            logger.warning(f"[HistoryStore] No messages for {key}")
            return []
        if n <= 0:
            return []
        subset = list(buf)[-n:]
        logger.info(f"[HistoryStore] Retrieved {len(subset)} messages for {key}")
        return subset

    # === Shared utilities for all commands ===

    @classmethod
    def get_recent_messages(
        cls,
        c: Context,
        n: int,
        exclude_current: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        Retrieves the last 'n' messages (users + bot).
        Excludes current message by timestamp (more reliable than by text).
        Returns a list of (name, text) without timestamp.
        """
        if n <= 0:
            return []

        target = max(n * 3, n + 1)
        all_msgs = cls.get_recent_for_context(c, target)

        if exclude_current:
            current_ts = _extract_timestamp_from_context(c)
            # Exclude by timestamp (100ms tolerance)
            all_msgs = [
                (name, text, ts) for (name, text, ts) in all_msgs
                if abs(ts - current_ts) > 100
            ]

        # Keep only last n after filtering, return without timestamp
        result = all_msgs[-n:] if all_msgs else []
        return [(name, text) for (name, text, _) in result]

    @staticmethod
    def format_history_lines(msgs: List[Tuple[str, str]], sep: str = "\n") -> str:
        """
        Formats a list (name, text) into lines 'name: text'.
        """
        if not msgs:
            return "(empty)"
        return sep.join(f"{name}: {text}" for (name, text) in msgs)


class HistoryRecorder(Command):
    """
    'Passive' command that records all incoming messages into HistoryStore.
    """
    def __init__(self, max_messages_per_chat: int = 300):
        super().__init__()
        HistoryStore._store = defaultdict(lambda: deque(maxlen=max_messages_per_chat))

    async def handle(self, c: Context):
        txt = getattr(c.message, "text", None)
        if txt:
            logger.debug(f"[HistoryRecorder] Message received: {txt}")
            HistoryStore.add(c, txt)
        else:
            logger.debug("[HistoryRecorder] No text to record.")