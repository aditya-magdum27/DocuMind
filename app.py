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
from memory_store import create_session, list_sessions, get_messages, delete_session
from rag_query import run_query

PERSIST_DIR = "./chroma_db"
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="DocuMind", layout="centered")
st.title("DocuMind")
st.caption("Multi-user RAG with OCR routing, hybrid retrieval, and saved chat history")

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

        # ---- chat sessions (stored in the database) ----
        sel_key = f"session_id::{user_id}::{document_id}"
        sessions = list_sessions(user_id, document_id)
        if not sessions:
            create_session(user_id, document_id)
            sessions = list_sessions(user_id, document_id)

        col_pick, col_new, col_del = st.columns([4, 1, 1])

        # Buttons are handled BEFORE the selectbox is drawn, so we are
        # allowed to change its value in session_state here.
        with col_new:
            if st.button("New chat"):
                if sessions[0]["message_count"] == 0:
                    new_id = sessions[0]["id"]  # latest chat is empty, reuse it
                else:
                    new_id = create_session(user_id, document_id)
                st.session_state[sel_key] = new_id
                st.rerun()
        with col_del:
            if st.button("Delete"):
                current = st.session_state.get(sel_key, sessions[0]["id"])
                delete_session(user_id, current)
                st.session_state.pop(sel_key, None)
                st.rerun()

        session_ids = [s["id"] for s in sessions]
        labels = {
            s["id"]: f"{s['title'] or 'New chat'}  ({s['updated_at'][:16].replace('T', ' ')})"
            for s in sessions
        }
        if st.session_state.get(sel_key) not in session_ids:
            st.session_state[sel_key] = session_ids[0]
        with col_pick:
            session_id = st.selectbox(
                "Chat", session_ids, format_func=lambda sid: labels[sid], key=sel_key
            )

        # ---- show the FULL saved chat, with sources ----
        for m in get_messages(user_id, session_id):
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if m["sources"]:
                    with st.expander("Sources"):
                        for s in m["sources"]:
                            st.write(f"- {s['file']}, page {s['page']}")

        query = st.chat_input("Ask a question about this document")
        if query:
            st.chat_message("user").write(query)
            with st.spinner("Thinking..."):
                run_query(user_id, document_id, query, PERSIST_DIR, session_id=session_id)
            # Rerun so the chat is redrawn from the database (and the chat
            # title / order in the picker updates).
            st.rerun()
