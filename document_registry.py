"""Document registry, now a table in documind.db (see db.py) instead of
documents.json. Function names and return shapes are unchanged, so
ingest.py and app.py keep working as before."""

from datetime import datetime, timezone

from db import db_conn


def add_document(document_id: str, user_id: str, source_name: str, num_pages: int, ocr_page_count: int) -> None:
    with db_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(document_id, user_id, source_name, num_pages, ocr_page_count, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (document_id, user_id, source_name, num_pages, ocr_page_count,
             datetime.now(timezone.utc).isoformat()),
        )


def list_documents(user_id: str) -> dict:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT document_id, source_name, num_pages, ocr_page_count, uploaded_at "
            "FROM documents WHERE user_id = ? ORDER BY uploaded_at",
            (user_id,),
        ).fetchall()
    return {
        r["document_id"]: {
            "user_id": user_id,
            "source_name": r["source_name"],
            "num_pages": r["num_pages"],
            "ocr_page_count": r["ocr_page_count"],
            "uploaded_at": r["uploaded_at"],
        }
        for r in rows
    }
