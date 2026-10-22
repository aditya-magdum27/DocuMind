"""SQLite setup for DocuMind: chat sessions, messages, and the document
registry all live in one file (documind.db). sqlite3 is built into Python,
so no new dependency is needed."""

import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.getenv("DOCUMIND_DB", "./documind.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id    TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    source_name    TEXT NOT NULL,
    num_pages      INTEGER,
    ocr_page_count INTEGER,
    uploaded_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    document_id TEXT NOT NULL,
    title       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sess_user_doc ON chat_sessions(user_id, document_id);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT NOT NULL,
    sources_json TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);
"""


@contextmanager
def db_conn():
    """Opens a connection, commits on success, rolls back on error, and
    always closes. One short-lived connection per call keeps this safe
    with Streamlit (which reruns the script in different threads)."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db_conn() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)


# Idempotent (CREATE ... IF NOT EXISTS), so it is safe to run on every import.
# This way the CLI scripts and the Streamlit app both get the tables.
init_db()
