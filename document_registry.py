"""Tiny JSON-backed registry so the UI can list which documents exist
per user, without having to scan the whole Chroma store to find out
what's been ingested."""

import json
import os
from datetime import datetime
from threading import Lock

REGISTRY_PATH = "./documents.json"
_lock = Lock()


def _load() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def add_document(document_id: str, user_id: str, source_name: str, num_pages: int, ocr_page_count: int) -> None:
    with _lock:
        data = _load()
        data[document_id] = {
            "user_id": user_id,
            "source_name": source_name,
            "num_pages": num_pages,
            "ocr_page_count": ocr_page_count,
            "uploaded_at": datetime.utcnow().isoformat(),
        }
        _save(data)


def list_documents(user_id: str) -> dict:
    data = _load()
    return {doc_id: meta for doc_id, meta in data.items() if meta["user_id"] == user_id}
