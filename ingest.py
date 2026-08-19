import os
import sys
import argparse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import fitz  # PyMuPDF

from ocr_router import get_page_text
from document_registry import add_document


def load_pdf_as_documents(pdf_path: str) -> list:
    """Build one Document per page, routing each page through the OCR
    graph (classify_page -> [run_ocr] -> END). Pages with a real text
    layer go through PyMuPDF only and never touch the OCR model - TrOCR
    only runs on pages that come back empty (scanned/handwritten)."""
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    documents = []
    for page_num in range(num_pages):
        text, ocr_used = get_page_text(pdf_path, page_num)
        if not text.strip():
            continue  # genuinely empty page, nothing to index
        documents.append(
            Document(
                page_content=text,
                metadata={"page": page_num, "ocr_used": ocr_used},
            )
        )
    return documents


def main():
    parser = argparse.ArgumentParser(description="Ingest a single PDF into the shared Chroma store.")
    parser.add_argument("--file", required=True, help="Path to the PDF to ingest")
    parser.add_argument("--user-id", required=True, help="Owning user's UUID")
    parser.add_argument("--document-id", required=True, help="This document's UUID")
    parser.add_argument("--source-name", required=True, help="Original filename, for display/citations")
    parser.add_argument("--persist-dir", default="./chroma_db")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400
    )

    # Each page is routed through the OCR graph individually: pages with
    # a real text layer are handled by PyMuPDF alone, pages that come
    # back empty (scanned/handwritten) fall through to TrOCR.
    page_documents = load_pdf_as_documents(args.file)
    chunks = splitter.split_documents(page_documents)

    if not chunks:
        print("ERROR: no extractable text found in this PDF", file=sys.stderr)
        sys.exit(1)

    ocr_page_count = sum(1 for d in page_documents if d.metadata.get("ocr_used"))

    # user_id and document_id are what make multi-tenant isolation possible:
    # rag_query.py filters on these at read time so one user's chat can
    # never surface another user's chunks.
    for chunk in chunks:
        chunk.metadata["source_file"] = args.source_name
        chunk.metadata["user_id"] = args.user_id
        chunk.metadata["document_id"] = args.document_id

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # This directory is shared by every user and every document.
    # Isolation happens entirely through metadata filtering at query time,
    # so ingestion only ever adds to the store - never clears it.
    vectorstore = Chroma(
        persist_directory=args.persist_dir,
        embedding_function=embeddings
    )
    vectorstore.add_documents(chunks)

    add_document(args.document_id, args.user_id, args.source_name, len(page_documents), ocr_page_count)

    print(f"OK: {len(chunks)} chunks stored for document {args.document_id} "
          f"({ocr_page_count}/{len(page_documents)} pages needed OCR)")


if __name__ == "__main__":
    main()
