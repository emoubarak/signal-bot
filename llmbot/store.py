"""SQLite message store.

One row per Signal message (users and bot alike), keyed by (chat, timestamp,
author): that is Signal's own message identity, and what a quote points to.
"""
from __future__ import annotations

import json
import unicodedata
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

KIND_TEXT = "text"
KIND_VOICE = "voice"
KIND_IMAGE = "image"
KIND_BOT = "bot"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    chat_id      TEXT    NOT NULL,
    ts           INTEGER NOT NULL,
    author_id    TEXT    NOT NULL,
    author_name  TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    text         TEXT    NOT NULL DEFAULT '',
    reply_to_ts  INTEGER,
    attachments  TEXT    NOT NULL DEFAULT '[]',
    PRIMARY KEY (chat_id, ts, author_id)
);
CREATE INDEX IF NOT EXISTS messages_chat_ts ON messages (chat_id, ts);

CREATE TABLE IF NOT EXISTS usage (
    ts       INTEGER NOT NULL,
    chat_id  TEXT    NOT NULL,
    user_id  TEXT    NOT NULL,
    kind     TEXT    NOT NULL,
    model    TEXT    NOT NULL,
    cost     REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS usage_user_ts ON usage (user_id, kind, ts);

CREATE TABLE IF NOT EXISTS reminders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      TEXT    NOT NULL,
    author_id    TEXT    NOT NULL,
    author_name  TEXT    NOT NULL,
    due_ts       INTEGER NOT NULL,
    message      TEXT    NOT NULL,
    created_ts   INTEGER NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    done         INTEGER NOT NULL DEFAULT 0,
    failed       INTEGER NOT NULL DEFAULT 0,
    next_try_ts  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS reminders_due ON reminders (done, due_ts);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _fold(text: str) -> str:
    """Lower-case without accents, so "resto" finds "Resto" and "été" finds "ete"."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


@dataclass
class StoredMessage:
    chat_id: str
    ts: int
    author_id: str
    author_name: str
    kind: str = KIND_TEXT
    text: str = ""
    reply_to_ts: int | None = None
    # [{"path": "...", "content_type": "image/jpeg"}, ...]
    attachments: list[dict] = field(default_factory=list)

    @property
    def is_bot(self) -> bool:
        return self.kind == KIND_BOT

    def images(self) -> list[dict]:
        return [a for a in self.attachments if str(a.get("content_type", "")).startswith("image/")]


def _row_to_message(row: sqlite3.Row) -> StoredMessage:
    return StoredMessage(
        chat_id=row["chat_id"],
        ts=row["ts"],
        author_id=row["author_id"],
        author_name=row["author_name"],
        kind=row["kind"],
        text=row["text"],
        reply_to_ts=row["reply_to_ts"],
        attachments=json.loads(row["attachments"] or "[]"),
    )


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Columns added after a database was first created."""
        have = {r["name"] for r in self._db.execute("PRAGMA table_info(reminders)")}
        for column in ("failed", "next_try_ts"):
            if column not in have:
                self._db.execute(f"ALTER TABLE reminders ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
                if column == "failed":
                    # reminders the old version had given up on (5 attempts) must not come back to life
                    self._db.execute("UPDATE reminders SET failed=1 WHERE done=0 AND attempts>=5")
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # ── messages ────────────────────────────────────────────────────────────

    def add(self, m: StoredMessage) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO messages VALUES (?,?,?,?,?,?,?,?)",
            (m.chat_id, m.ts, m.author_id, m.author_name, m.kind, m.text, m.reply_to_ts,
             json.dumps(m.attachments)),
        )
        self._db.commit()

    def update_text(self, chat_id: str, ts: int, author_id: str, text: str) -> bool:
        cur = self._db.execute(
            "UPDATE messages SET text=? WHERE chat_id=? AND ts=? AND author_id=?",
            (text, chat_id, ts, author_id),
        )
        self._db.commit()
        return cur.rowcount > 0

    def update_message(self, chat_id: str, ts: int, author_id: str, text: str, attachments: list[dict]) -> bool:
        cur = self._db.execute(
            "UPDATE messages SET text=?, attachments=? WHERE chat_id=? AND ts=? AND author_id=?",
            (text, json.dumps(attachments), chat_id, ts, author_id),
        )
        self._db.commit()
        return cur.rowcount > 0

    def delete_message(self, chat_id: str, ts: int, author_ids: set[str]) -> list[dict] | None:
        """Remove a message its author deleted for everyone. Returns its attachments (the caller removes the
        files), or None when no message of one of `author_ids` is stored at that time."""
        for author_id in author_ids:
            row = self._db.execute(
                "SELECT attachments FROM messages WHERE chat_id=? AND ts=? AND author_id=?", (chat_id, ts, author_id)
            ).fetchone()
            if row is None:
                continue
            self._db.execute("DELETE FROM messages WHERE chat_id=? AND ts=? AND author_id=?", (chat_id, ts, author_id))
            self._db.commit()
            return json.loads(row["attachments"] or "[]")
        return None

    def has_reply(self, chat_id: str, ts: int, bot_id: str) -> bool:
        """Whether the bot (author `bot_id`) already answered the message at `ts`, with text or with an image."""
        row = self._db.execute(
            "SELECT 1 FROM messages WHERE chat_id=? AND reply_to_ts=? AND author_id=? LIMIT 1", (chat_id, ts, bot_id)
        ).fetchone()
        return row is not None

    def has_posted_in(self, author_id: str, chat_ids: set[str]) -> bool:
        """Whether this author wrote in one of these chats (a member of a group, as far as the bot has seen)."""
        if not chat_ids:
            return False
        marks = ",".join("?" * len(chat_ids))
        row = self._db.execute(
            f"SELECT 1 FROM messages WHERE author_id=? AND chat_id IN ({marks}) LIMIT 1", (author_id, *chat_ids)
        ).fetchone()
        return row is not None

    def get(self, chat_id: str, ts: int) -> StoredMessage | None:
        row = self._db.execute(
            "SELECT * FROM messages WHERE chat_id=? AND ts=? LIMIT 1", (chat_id, ts)
        ).fetchone()
        return _row_to_message(row) if row else None

    def recent(
        self,
        chat_id: str,
        limit: int,
        *,
        before_ts: int | None = None,
        after_ts: int | None = None,
    ) -> list[StoredMessage]:
        """Last `limit` messages of a chat, oldest first."""
        sql = "SELECT * FROM messages WHERE chat_id=?"
        params: list = [chat_id]
        if before_ts is not None:
            sql += " AND ts<?"
            params.append(before_ts)
        if after_ts is not None:
            sql += " AND ts>?"
            params.append(after_ts)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self._db.execute(sql, params).fetchall()
        return [_row_to_message(r) for r in reversed(rows)]

    def conversation_thread(self, chat_id: str, ts: int, max_len: int = 12) -> list[StoredMessage]:
        """The reply chain ending at `ts`, followed only through exchanges with the
        bot (user → bot → user …): a reply to another person's message is not a
        conversation turn but a target. Oldest first."""
        current = self.get(chat_id, ts)
        if current is None:
            return []
        chain = [current]
        while current.reply_to_ts is not None and len(chain) < max_len:
            prev = self.get(chat_id, current.reply_to_ts)
            if prev is None or not (prev.is_bot or current.is_bot) or prev.ts in {m.ts for m in chain}:
                break
            chain.insert(0, prev)
            current = prev
        return chain

    def author_name(self, chat_id: str, author_id: str) -> str | None:
        row = self._db.execute(
            "SELECT author_name FROM messages WHERE chat_id=? AND author_id=? ORDER BY ts DESC LIMIT 1",
            (chat_id, author_id),
        ).fetchone()
        return row["author_name"] if row else None

    def names_by_author(self, chat_id: str) -> dict[str, list[str]]:
        """Every display name each author of the chat had, the current one first."""
        out: dict[str, list[str]] = {}
        for row in self._db.execute(
                "SELECT author_id, author_name, MAX(ts) AS last FROM messages WHERE chat_id=? AND kind != 'bot' "
                "GROUP BY author_id, author_name ORDER BY last DESC", (chat_id,)):
            out.setdefault(row["author_id"], []).append(row["author_name"])
        return out

    def prune(self, keep_per_chat: int = 2000) -> None:
        chats = [r["chat_id"] for r in self._db.execute("SELECT DISTINCT chat_id FROM messages")]
        for chat in chats:
            self._db.execute(
                "DELETE FROM messages WHERE chat_id=? AND ts < COALESCE((SELECT ts FROM messages "
                "WHERE chat_id=? ORDER BY ts DESC LIMIT 1 OFFSET ?), 0)",
                (chat, chat, keep_per_chat),
            )
        self._db.commit()

    # ── search ──────────────────────────────────────────────────────────────

    def search(self, chat_id: str, query: str, *, limit: int = 20, after_ts: int | None = None,
               before_ts: int | None = None, scan: int = 5000) -> list[StoredMessage]:
        """Messages of a chat containing every word of `query`, case and accents ignored, newest `limit`
        in reading order. Looks through the last `scan` messages: plenty for a group, and a LIKE would not
        fold accents."""
        words = [w for w in _fold(query).split() if w]
        if not words:
            return []
        hits = [m for m in reversed(self.recent(chat_id, scan, before_ts=before_ts, after_ts=after_ts))
                if (t := _fold(m.text)) and all(w in t for w in words)]
        return list(reversed(hits[:limit]))

    # ── reminders ───────────────────────────────────────────────────────────

    def add_reminder(self, chat_id: str, author_id: str, author_name: str, due_ts: int, message: str,
                     now_ts: int) -> int:
        cur = self._db.execute(
            "INSERT INTO reminders (chat_id, author_id, author_name, due_ts, message, created_ts) VALUES (?,?,?,?,?,?)",
            (chat_id, author_id, author_name, due_ts, message, now_ts))
        self._db.commit()
        return int(cur.lastrowid)

    def pending_reminders(self, chat_id: str) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM reminders WHERE chat_id=? AND done=0 AND failed=0 ORDER BY due_ts", (chat_id,)).fetchall()

    def due_reminders(self, now_ts: int) -> list[sqlite3.Row]:
        """Reminders to post now: due, not sent, not given up, and past their pause after a failed attempt."""
        return self._db.execute(
            "SELECT * FROM reminders WHERE done=0 AND failed=0 AND due_ts<=? AND next_try_ts<=? ORDER BY due_ts",
            (now_ts, now_ts)).fetchall()

    def finish_reminder(self, reminder_id: int) -> None:
        self._db.execute("UPDATE reminders SET done=1 WHERE id=?", (reminder_id,))
        self._db.commit()

    def reminder_failed(self, reminder_id: int, retry_at: int | None) -> None:
        """One more failed attempt: try again from `retry_at`, or for good (`None`: marked failed, no longer
        waiting)."""
        if retry_at is None:
            self._db.execute("UPDATE reminders SET attempts=attempts+1, failed=1 WHERE id=?", (reminder_id,))
        else:
            self._db.execute("UPDATE reminders SET attempts=attempts+1, next_try_ts=? WHERE id=?",
                             (retry_at, reminder_id))
        self._db.commit()

    def cancel_reminder(self, chat_id: str, reminder_id: int) -> bool:
        cur = self._db.execute("UPDATE reminders SET done=1 WHERE id=? AND chat_id=? AND done=0", (reminder_id, chat_id))
        self._db.commit()
        return cur.rowcount > 0

    # ── usage / rate limiting ───────────────────────────────────────────────

    def add_usage(self, chat_id: str, user_id: str, kind: str, model: str, cost: float) -> int:
        cur = self._db.execute(
            "INSERT INTO usage VALUES (?,?,?,?,?,?)",
            (int(time.time() * 1000), chat_id, user_id, kind, model, cost or 0.0),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def update_usage(self, rowid: int, model: str, cost: float) -> None:
        """Fill in a usage reserved before the call (see add_usage) with what it really cost."""
        self._db.execute("UPDATE usage SET model=?, cost=? WHERE rowid=?", (model, cost or 0.0, rowid))
        self._db.commit()

    def count_usage(self, user_id: str, kind: str, since_ts: int) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM usage WHERE user_id=? AND kind=? AND ts>=?",
            (user_id, kind, since_ts),
        ).fetchone()
        return int(row["n"])

    def cost_since(self, since_ts: int, chat_id: str | None = None) -> float:
        if chat_id is None:
            row = self._db.execute("SELECT COALESCE(SUM(cost),0) AS c FROM usage WHERE ts>=?", (since_ts,)).fetchone()
        else:
            row = self._db.execute(
                "SELECT COALESCE(SUM(cost),0) AS c FROM usage WHERE ts>=? AND chat_id=?", (since_ts, chat_id)
            ).fetchone()
        return float(row["c"])

    # ── settings ────────────────────────────────────────────────────────────

    def get_setting(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str | None) -> None:
        if value is None:
            self._db.execute("DELETE FROM settings WHERE key=?", (key,))
        else:
            self._db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, value))
        self._db.commit()

    # ── context clearing (/clear) ────────────────────────────────────────────

    def clear_context(self, chat_id: str, ts: int) -> int:
        """Make the bot forget everything said in `chat_id` before `ts`.

        The messages stay in the database (summaries, history and cost are
        unaffected); the model simply stops being shown them, and `get_history`
        stops reaching past this point. Returns how many messages were hidden."""
        self.set_setting(self._clear_key(chat_id), str(ts))
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE chat_id=? AND ts<?", (chat_id, ts)
        ).fetchone()
        return int(row["n"])

    def context_start(self, chat_id: str) -> int | None:
        """The timestamp of the last /clear in this chat, or None."""
        value = self.get_setting(self._clear_key(chat_id))
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None

    @staticmethod
    def _clear_key(chat_id: str) -> str:
        return f"context_cleared_at:{chat_id}"
