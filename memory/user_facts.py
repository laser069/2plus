import sqlite3
import time
from pathlib import Path

from config.settings import SQLITE_PATH, FACTS_MAX_CHARS


def _conn() -> sqlite3.Connection:
    Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    con.execute(
        "CREATE TABLE IF NOT EXISTS user_facts "
        "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
    )
    con.commit()
    return con


def upsert(key: str, value: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _conn() as con:
        con.execute(
            "INSERT INTO user_facts(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, ts),
        )


def get(key: str) -> str | None:
    with _conn() as con:
        row = con.execute("SELECT value FROM user_facts WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def get_all() -> dict[str, str]:
    with _conn() as con:
        rows = con.execute("SELECT key, value FROM user_facts").fetchall()
    return {k: v for k, v in rows}


def delete(key: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM user_facts WHERE key=?", (key,))


def get_relevant(query: str, budget: int = FACTS_MAX_CHARS) -> str:
    """Return facts whose key/value overlap with query words; fall back to all facts.
    Output is a formatted string, truncated to budget chars."""
    all_facts = get_all()
    if not all_facts:
        return ""

    query_words = set(query.lower().split())
    scored: list[tuple[int, str, str]] = []
    for k, v in all_facts.items():
        overlap = len(query_words & set((k + " " + v).lower().split()))
        scored.append((overlap, k, v))

    scored.sort(key=lambda x: -x[0])
    lines = [f"- {k}: {v}" for _, k, v in scored]
    block = "\n".join(lines)
    return block[:budget]
