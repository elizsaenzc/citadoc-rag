"""Persistent conversations (SQLite).

Each question and answer is stored with its sources, so a conversation can be
reopened later and follow-up questions can use the previous turns as context.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.models import ConversationDetail, ConversationSummary, StoredMessage

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    payload          TEXT,             -- JSON: sources, steps, grounded... (assistant only)
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation_id, id);
"""


class ConversationNotFoundError(LookupError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)

    def create(self, title: str) -> str:
        conversation_id = uuid.uuid4().hex[:12]
        now = _now()
        title = title.strip().replace("\n", " ")
        title = title if len(title) <= 60 else title[:57] + "…"
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now),
            )
        return conversation_id

    def exists(self, conversation_id: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return row is not None

    def add_message(
        self, conversation_id: str, role: str, content: str, payload: dict | None = None
    ) -> None:
        now = _now()
        encoded = json.dumps(payload, ensure_ascii=False) if payload else None
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages (conversation_id, role, content, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, encoded, now),
            )
            self._db.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
            )

    def history(self, conversation_id: str, limit: int = 6) -> list[tuple[str, str]]:
        """Last ``limit`` messages as (role, content), oldest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [(r["role"], r["content"]) for r in reversed(rows)]

    def list(self) -> list[ConversationSummary]:
        with self._lock:
            rows = self._db.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS messages "
                "FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id "
                "GROUP BY c.id ORDER BY c.updated_at DESC, c.created_at DESC"
            ).fetchall()
        return [ConversationSummary(**dict(r)) for r in rows]

    def get(self, conversation_id: str) -> ConversationDetail:
        with self._lock:
            conversation = self._db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)
            rows = self._db.execute(
                "SELECT role, content, payload, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        messages = [
            StoredMessage(
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
                **(json.loads(r["payload"]) if r["payload"] else {}),
            )
            for r in rows
        ]
        return ConversationDetail(**dict(conversation), messages=messages)

    def delete(self, conversation_id: str) -> None:
        with self._lock, self._db:
            deleted = self._db.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            ).rowcount
        if not deleted:
            raise ConversationNotFoundError(conversation_id)
