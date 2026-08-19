"""
Minimal Streamlit UI for the RAG pipeline.

Upload a PDF -> ingest it (OCR-routed page by page) -> pick which
document to chat with -> ask questions, with memory.

This file only orchestrates: all the real logic (OCR routing, hybrid
retrieval, memory) lives in ingest.py / rag_query.py / ocr_router.py /
memory_store.py, and is called the same way it would be from the CLI.
"""

import os
import uuid

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from ingest import load_pdf_as_documents
from document_registry import add_document, list_documents
from memory_store import load_history
from rag_query import run_query

PERSIST_DIR = "./chroma_db"
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="DocuMind", layout="centered")
st.title("DocuMind")
st.caption("Multi-user RAG with OCR routing, hybrid retrieval, and per-document memory")

# Stand-in for real auth: one text box for user id, scopes every query
# and every registry lookup, same as the isolation logic in rag_query.py.
user_id = st.sidebar.text_input("User ID", value="demo-user")

tab_upload, tab_chat = st.tabs(["Upload & Ingest", "Chat"])

# ---------------------------------------------------------------- upload
with tab_upload:
    st.subheader("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and st.button("Ingest"):
        document_id = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Reading pages and routing through OCR where needed..."):
            page_documents = load_pdf_as_documents(save_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
            chunks = splitter.split_documents(page_documents)

            for chunk in chunks:
                chunk.metadata["source_file"] = uploaded_file.name
                chunk.metadata["user_id"] = user_id
                chunk.metadata["document_id"] = document_id

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            vectorstore.add_documents(chunks)

            ocr_page_count = sum(1 for d in page_documents if d.metadata.get("ocr_used"))
            add_document(document_id, user_id, uploaded_file.name, len(page_documents), ocr_page_count)

        st.success(
            f"Ingested '{uploaded_file.name}' - {len(page_documents)} pages, "
            f"{ocr_page_count} needed OCR, {len(chunks)} chunks stored."
        )

# ------------------------------------------------------------------ chat
with tab_chat:
    st.subheader("Chat with a document")
    docs = list_documents(user_id)

    if not docs:
        st.info("No documents ingested yet for this user.")
    else:
        options = {f"{meta['source_name']} ({doc_id[:8]})": doc_id for doc_id, meta in docs.items()}
        choice = st.selectbox("Select a document", list(options.keys()))
        document_id = options[choice]

        history = load_history(user_id, document_id)
        for turn in history:
            st.chat_message("user").write(turn["question"])
            st.chat_message("assistant").write(turn["answer"])

        query = st.chat_input("Ask a question about this document")
        if query:
            st.chat_message("user").write(query)
            with st.spinner("Thinking..."):
                result = run_query(user_id, document_id, query, PERSIST_DIR)
            st.chat_message("assistant").write(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.write(f"- {s['file']}, page {s['page']}")
