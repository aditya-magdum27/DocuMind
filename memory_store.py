"""Chat storage + conversation memory, backed by SQLite (see db.py).

Two separate ideas live here:
  1. FULL history  -> every message is saved forever (get_messages, for the UI).
  2. Model context -> only the last MAX_TURNS turns are sent to the LLM
                      (load_history, for the prompt).

Every query checks user_id, same isolation idea as the Chroma filters in
rag_query.py: one user can never read or write another user's chat.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from db import db_conn

MAX_TURNS = 4            # how many past (question, answer) pairs go to the LLM
MAX_ANSWER_CHARS = 600   # long old answers are cut so they don't fill the prompt
TITLE_CHARS = 40


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- sessions
def create_session(user_id: str, document_id: str, title: Optional[str] = None) -> str:
    session_id = str(uuid.uuid4())
    now = _now()
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, user_id, document_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, user_id, document_id, title, now, now),
        )
    return session_id


def list_sessions(user_id: str, document_id: str) -> list:
    """Newest chat first. message_count lets the UI spot empty chats."""
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT s.id, s.title, s.created_at, s.updated_at, "
            "       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count "
            "FROM chat_sessions s "
            "WHERE s.user_id = ? AND s.document_id = ? "
            "ORDER BY s.updated_at DESC",
            (user_id, document_id),
        ).fetchall()
    return [dict(r) for r in rows]


def session_belongs_to(user_id: str, session_id: str, document_id: Optional[str] = None) -> bool:
    query = "SELECT 1 FROM chat_sessions WHERE id = ? AND user_id = ?"
    params = [session_id, user_id]
    if document_id is not None:
        query += " AND document_id = ?"
        params.append(document_id)
    with db_conn() as conn:
        return conn.execute(query, params).fetchone() is not None


def delete_session(user_id: str, session_id: str) -> None:
    # messages are removed automatically (ON DELETE CASCADE)
    with db_conn() as conn:
        conn.execute(
            "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )


# ---------------------------------------------------------------- messages
def get_messages(user_id: str, session_id: str) -> list:
    """FULL history of one chat, oldest first. Used to draw the chat in the UI."""
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT m.role, m.content, m.sources_json, m.created_at "
            "FROM messages m JOIN chat_sessions s ON s.id = m.session_id "
            "WHERE m.session_id = ? AND s.user_id = ? "
            "ORDER BY m.id",
            (session_id, user_id),
        ).fetchall()

    messages = []
    for r in rows:
        messages.append({
            "role": r["role"],
            "content": r["content"],
            "sources": json.loads(r["sources_json"]) if r["sources_json"] else [],
            "created_at": r["created_at"],
        })
    return messages


def load_history(user_id: str, session_id: str, limit: int = MAX_TURNS) -> list:
    """Last `limit` (question, answer) turns, oldest first. Used for the
    LLM prompt only - the UI should use get_messages()."""
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT m.role, m.content "
            "FROM messages m JOIN chat_sessions s ON s.id = m.session_id "
            "WHERE m.session_id = ? AND s.user_id = ? "
            "ORDER BY m.id DESC LIMIT ?",
            (session_id, user_id, limit * 2),
        ).fetchall()

    rows = list(reversed(rows))  # back to oldest-first
    turns = []
    pending_question = None
    for r in rows:
        if r["role"] == "user":
            pending_question = r["content"]
        elif r["role"] == "assistant" and pending_question is not None:
            turns.append({"question": pending_question, "answer": r["content"]})
            pending_question = None
    return turns


def save_turn(user_id: str, session_id: str, question: str, answer: str,
              sources: Optional[list] = None) -> None:
    """Saves the user message and the assistant message together, in one
    transaction, so a chat never ends up with a question but no answer."""
    if not session_belongs_to(user_id, session_id):
        raise ValueError("Chat session not found for this user.")

    now = _now()
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
            (session_id, question, now),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, sources_json, created_at) "
            "VALUES (?, 'assistant', ?, ?, ?)",
            (session_id, answer, json.dumps(sources or []), now),
        )
        # first question becomes the chat title
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ?, "
            "title = COALESCE(title, ?) WHERE id = ?",
            (now, question.strip()[:TITLE_CHARS], session_id),
        )


def format_history(history: list) -> str:
    if not history:
        return ""
    lines = []
    for turn in history:
        answer = turn["answer"]
        if len(answer) > MAX_ANSWER_CHARS:
            answer = answer[:MAX_ANSWER_CHARS] + "..."
        lines.append(f"Q: {turn['question']}")
        lines.append(f"A: {answer}")
    return "\n".join(lines)
