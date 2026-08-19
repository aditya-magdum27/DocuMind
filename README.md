# DocuMind

Multi-user RAG system with OCR routing, hybrid retrieval, and per-document memory.

## What happens on ingest (`ingest.py`)

1. PDF is opened page by page (PyMuPDF).
2. Each page goes through `ocr_router.py`, a LangGraph graph:
   - **`classify_page`** node: tries normal PyMuPDF text extraction. If the
     extracted text is under ~20 characters, the page is flagged
     `needs_ocr = True` (this is the signature of a scanned/handwritten
     page - there's no text layer to pull from).
   - **conditional edge**: routes to `run_ocr` only if `needs_ocr` is
     True. Pages with a real text layer skip OCR entirely and go
     straight to `END`.
   - **`run_ocr`** node: renders the page as an image and runs it
     through **TrOCR** (`microsoft/trocr-base-handwritten`) to extract
     text.
3. All page texts (whichever path each took) are wrapped as LangChain
   `Document`s and chunked (`RecursiveCharacterTextSplitter`,
   chunk_size=1500, overlap=400) - page metadata carries through
   automatically.
4. Each chunk gets `user_id` and `document_id` tagged onto its metadata.
   This is the entire isolation mechanism: nothing is separated by
   folder or by collection, it's all enforced by metadata filtering
   at query time.
5. Chunks are embedded (`sentence-transformers/all-MiniLM-L6-v2`) and
   added to a single shared Chroma store.
6. The document is registered in `documents.json` (a tiny local
   registry) so the UI can list "which PDFs has this user uploaded"
   without scanning the whole vector store.

## What happens on query (`rag_query.py`)

1. Every read is filtered by `{"document_id": ..., "user_id": ...}` -
   this is the authorization boundary. User A can never retrieve user
   B's chunks, and a chat session can never pull chunks from a
   different document than the one selected.
2. **Hybrid retrieval**: runs both semantic search (Chroma similarity
   search over embeddings) and BM25 (keyword/lexical scoring) over the
   same filtered chunk set, takes the top 10 from each, and de-dupes
   by `(source_file, page)`.
3. **Memory** (`memory_store.py`): the last 4 (question, answer) turns
   for this `(user_id, document_id)` session are loaded from a small
   JSON file and prepended to the prompt as plain text, so the model
   can resolve follow-up references without treating prior answers as
   extra document facts.
4. The context + history + question go into a prompt template and are
   sent to Groq, constrained to answer only from the given context.
5. Top 5 unique sources (file + 1-indexed page number) are returned
   alongside the answer, and the turn is saved to memory.

## Tech stack

Python, PyMuPDF, LangChain, LangGraph, ChromaDB, BM25 (rank_bm25),
HuggingFace sentence-transformers (all-MiniLM-L6-v2), TrOCR
(microsoft/trocr-base-handwritten), Groq, Streamlit

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY

# CLI ingest
python ingest.py --file mydoc.pdf --user-id u1 --document-id d1 --source-name mydoc.pdf

# CLI query
python rag_query.py --user-id u1 --document-id d1 --query "What is this document about?"

# UI
streamlit run app.py
```

## Known limitations (honest scope)

- OCR routing uses a simple character-count heuristic (~20 chars), not
  a trained classifier.
- Memory is a plain rolling window (last 4 turns), no summarization or
  vector-based memory.
- `documents.json` is a flat-file registry, would move to a proper DB
  table in production.
