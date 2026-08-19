"""Simplest possible conversation memory: a JSON file per (user, document)
session holding the last few (question, answer) turns. No vector store, no
summarization - just a plain rolling window, so every line of this is easy
to explain in an interview.

This is functionally the same idea as LangChain's
ConversationBufferWindowMemory, hand-rolled so the mechanics stay visible.
"""

import json
import os
from threading import Lock

MEMORY_DIR = "./memory"
MAX_TURNS = 4  # how many past (question, answer) pairs we keep
_lock = Lock()


def _session_path(user_id: str, document_id: str) -> str:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    return os.path.join(MEMORY_DIR, f"{user_id}__{document_id}.json")


def load_history(user_id: str, document_id: str) -> list:
    path = _session_path(user_id, document_id)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_turn(user_id: str, document_id: str, question: str, answer: str) -> None:
    with _lock:
        path = _session_path(user_id, document_id)
        history = load_history(user_id, document_id)
        history.append({"question": question, "answer": answer})
        history = history[-MAX_TURNS:]  # keep only the last MAX_TURNS
        with open(path, "w") as f:
            json.dump(history, f, indent=2)


def format_history(history: list) -> str:
    if not history:
        return ""
    lines = []
    for turn in history:
        lines.append(f"Q: {turn['question']}")
        lines.append(f"A: {turn['answer']}")
    return "\n".join(lines)
