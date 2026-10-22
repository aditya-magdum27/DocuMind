"""One-time import of the OLD json files into documind.db.

  documents.json          -> documents table
  memory/<user>__<doc>.json -> one chat session each (old last-4 turns)

Safe to run twice: documents use INSERT OR IGNORE, and a (user, document)
that already has a chat session is skipped. Old files are NOT deleted.

Run:  python migrate_to_db.py
"""

import glob
import json
import os
from datetime import datetime, timezone

from db import db_conn
from memory_store import create_session, save_turn, list_sessions

REGISTRY_PATH = "./documents.json"
MEMORY_DIR = "./memory"


def migrate_documents() -> int:
    if not os.path.exists(REGISTRY_PATH):
        return 0
    with open(REGISTRY_PATH, "r") as f:
        data = json.load(f)
    count = 0
    with db_conn() as conn:
        for doc_id, m in data.items():
            cur = conn.execute(
                "INSERT OR IGNORE INTO documents "
                "(document_id, user_id, source_name, num_pages, ocr_page_count, uploaded_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, m["user_id"], m["source_name"], m.get("num_pages"),
                 m.get("ocr_page_count"),
                 m.get("uploaded_at") or datetime.now(timezone.utc).isoformat()),
            )
            count += cur.rowcount
    return count


def migrate_chats() -> int:
    count = 0
    for path in glob.glob(os.path.join(MEMORY_DIR, "*.json")):
        name = os.path.splitext(os.path.basename(path))[0]
        if "__" not in name:
            continue
        user_id, document_id = name.rsplit("__", 1)  # document_id is a uuid, no "__" in it
        if list_sessions(user_id, document_id):
            continue  # already has chats in the DB
        with open(path, "r") as f:
            turns = json.load(f)
        if not turns:
            continue
        session_id = create_session(user_id, document_id)
        for t in turns:
            save_turn(user_id, session_id, t["question"], t["answer"], [])
        count += 1
    return count


if __name__ == "__main__":
    print(f"documents imported: {migrate_documents()}")
    print(f"chats imported:     {migrate_chats()}")
