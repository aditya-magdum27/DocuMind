# DocuMind

A RAG (Retrieval-Augmented Generation) system for **scanned and digital PDFs**. You give it one
PDF, then ask questions about it and get answers with page-number sources.
It uses OCR routing for scanned pages, hybrid retrieval (semantic + keyword), and chat memory.

**Scope:** one document at a time, PDF files only. It is not a multi-user or multi-document
system, and it does not handle images, audio or other file types.

## What happens on ingest (`ingest.py`)

1. The PDF is opened page by page (PyMuPDF).
2. Each page goes through `ocr_router.py`, a LangGraph graph:
   - **`classify_page`** node: tries normal PyMuPDF text extraction. If the
     extracted text is under ~20 characters, the page is flagged
     `needs_ocr = True` (this is the signature of a scanned or handwritten
     page - there is no text layer to pull from).
   - **conditional edge**: routes to `run_ocr` only if `needs_ocr` is
     True. Pages with a real text layer skip OCR entirely and go
     straight to `END`.
   - **`run_ocr`** node: renders the page as an image and runs it
     through **TrOCR** (`microsoft/trocr-base-handwritten`) to extract
     text.
3. All page texts (whichever path each page took) are wrapped as LangChain
   `Document`s and chunked (`RecursiveCharacterTextSplitter`,
   chunk_size=1500, overlap=400). Page metadata carries through
   automatically, so every chunk knows its source file and page.
4. Chunks are embedded (`sentence-transformers/all-MiniLM-L6-v2`) and
   stored in a local Chroma vector store.

## What happens on query (`rag_query.py`)

1. **Hybrid retrieval**: runs both semantic search (Chroma similarity
   search over embeddings) and BM25 (keyword scoring) over the document's
   chunks, takes the top 10 from each, and de-dupes
   by `(source_file, page)`.
2. **Memory** (`memory_store.py`): the last 4 (question, answer) turns
   of the chat are loaded from a small JSON file and added to the prompt
   as plain text, so the model can understand follow-up questions
   without treating earlier answers as extra document facts.
3. The context, chat history and question go into a prompt template and are
   sent to Groq. The model is told to answer only from the given context.
4. The top 5 unique sources (file + 1-indexed page number) are returned
   with the answer, and the turn is saved to memory.

## Tech stack

Python, PyMuPDF, LangChain, LangGraph, ChromaDB, BM25 (rank_bm25),
HuggingFace sentence-transformers (all-MiniLM-L6-v2), TrOCR
(microsoft/trocr-base-handwritten), Groq, Streamlit

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY

# CLI ingest
python ingest.py --file mydoc.pdf --source-name mydoc.pdf

# CLI query
python rag_query.py --query "What is this document about?"

# UI
streamlit run app.py
```

Keep your real key only in `.env`. Never commit `.env` or put a key in `.env.example`.

## Known limitations (honest scope)

- OCR routing uses a simple character-count heuristic (~20 chars), not
  a trained classifier. A page with a little real text and a scanned body
  can be missed.
- TrOCR (`trocr-base-handwritten`) is built for single text lines, so
  dense or multi-column scanned pages may give weaker text.
- Chat memory is a plain rolling window (last 4 turns) stored in a JSON file,
  with no summarization or vector-based memory. Storing chat history and
  context in a database is a planned improvement.
- Answers are limited to the text found in the PDF. Charts, photos and
  diagrams inside the PDF are not understood.
