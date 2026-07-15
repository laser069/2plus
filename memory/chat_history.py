from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from config.settings import SQLITE_PATH


def _conn() -> sqlite3.Connection:
    Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    con.execute(
        "CREATE TABLE IF NOT EXISTS chat_messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT NOT NULL, "
        "role TEXT NOT NULL, "
        "content TEXT NOT NULL, "
        "citations TEXT, "
        "meta TEXT, "
        "ts TEXT NOT NULL"
        ")"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_session "
        "ON chat_messages(session_id, id)"
    )
    con.commit()
    return con


def save_message(
    session_id: str,
    role: str,
    content: str,
    citations: list[str] | None = None,
    meta: dict | None = None,
) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _conn() as con:
        con.execute(
            "INSERT INTO chat_messages(session_id, role, content, citations, meta, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                role,
                content,
                json.dumps(citations) if citations else None,
                json.dumps(meta) if meta else None,
                ts,
            ),
        )


def load_session(session_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT role, content, citations, meta FROM chat_messages "
            "WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    messages = []
    for role, content, citations_raw, meta_raw in rows:
        msg: dict = {"role": role, "content": content}
        if citations_raw:
            try:
                msg["citations"] = json.loads(citations_raw)
            except Exception:
                pass
        if meta_raw:
            try:
                msg["meta"] = json.loads(meta_raw)
            except Exception:
                pass
        messages.append(msg)
    return messages


def delete_session(session_id: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
