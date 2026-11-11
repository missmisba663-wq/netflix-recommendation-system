"""SQLite persistence layer: per-user watchlist and like/dislike feedback."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    username   TEXT NOT NULL,
    show_id    TEXT NOT NULL,
    added_at   TEXT NOT NULL,
    PRIMARY KEY (username, show_id)
);
CREATE TABLE IF NOT EXISTS feedback (
    username   TEXT NOT NULL,
    show_id    TEXT NOT NULL,
    liked      INTEGER NOT NULL,          -- 1 = like, 0 = dislike
    updated_at TEXT NOT NULL,
    PRIMARY KEY (username, show_id)
);
"""


@contextmanager
def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_to_watchlist(user: str, show_id: str) -> None:
    with connect() as con:
        con.execute("INSERT OR IGNORE INTO watchlist VALUES (?, ?, ?)", (user, show_id, _now()))


def remove_from_watchlist(user: str, show_id: str) -> None:
    with connect() as con:
        con.execute("DELETE FROM watchlist WHERE username=? AND show_id=?", (user, show_id))


def get_watchlist(user: str) -> list[str]:
    with connect() as con:
        rows = con.execute(
            "SELECT show_id FROM watchlist WHERE username=? ORDER BY added_at DESC", (user,)
        ).fetchall()
    return [r[0] for r in rows]


def set_feedback(user: str, show_id: str, liked: bool) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?) "
            "ON CONFLICT(username, show_id) DO UPDATE SET liked=excluded.liked, updated_at=excluded.updated_at",
            (user, show_id, int(liked), _now()),
        )


def clear_feedback(user: str, show_id: str) -> None:
    with connect() as con:
        con.execute("DELETE FROM feedback WHERE username=? AND show_id=?", (user, show_id))


def get_feedback(user: str) -> dict[str, bool]:
    with connect() as con:
        rows = con.execute("SELECT show_id, liked FROM feedback WHERE username=?", (user,)).fetchall()
    return {sid: bool(l) for sid, l in rows}
