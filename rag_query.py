import json
import argparse

import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from rank_bm25 import BM25Okapi

from memory_store import load_history, save_turn, format_history

load_dotenv()

_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def run_query(user_id: str, document_id: str, query: str, persist_dir: str = "./chroma_db") -> dict:
    """Runs one question against one document: hybrid retrieval
    (semantic + BM25) scoped to (user_id, document_id), plus rolling
    conversation memory for that session. Returns {"answer", "sources"}."""

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=_get_embeddings()
    )

    # Authorization boundary: every read below is scoped to this
    # document_id AND this user_id. This is what stops user A from ever
    # retrieving user B's chunks, and stops a chat session from pulling
    # in chunks from a different document than the one selected.
    doc_filter = {
        "$and": [
            {"document_id": {"$eq": document_id}},
            {"user_id": {"$eq": user_id}}
        ]
    }

    chroma_data = vectorstore._collection.get(
        where=doc_filter,
        include=["documents", "metadatas"]
    )

    if not chroma_data["documents"]:
        return {
            "answer": "I don't know",
            "sources": [],
            "error": "No indexed content found for this document."
        }

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,
            "filter": doc_filter
        }
    )

    corpus = [text.split() for text in chroma_data["documents"]]
    bm25 = BM25Okapi(corpus)

    semantic_docs = retriever.invoke(query)

    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    top_n = np.argsort(scores)[::-1][:10]

    class Doc:
        pass

    bm25_docs = []
    for i in top_n:
        text = chroma_data["documents"][i]
        meta = chroma_data["metadatas"][i]

        doc = Doc()
        doc.page_content = text
        doc.metadata = meta

        bm25_docs.append(doc)

    # Combine both retrieval methods, then de-dupe by (source, page) so
    # the same chunk found by both semantic and keyword search isn't
    # counted/shown twice.
    combined_docs = semantic_docs + bm25_docs
    seen = set()
    unique_docs = []

    for doc in combined_docs:
        key = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page")
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    context = "\n\n".join(doc.page_content for doc in unique_docs[:10])

    # Rolling-window memory: last few turns of this (user, document)
    # session are prepended as plain text so the model can resolve
    # follow-up references ("it", "that section") - not treated as
    # extra facts, only as conversational context.
    history = load_history(user_id, document_id)
    history_text = format_history(history)

    prompt = PromptTemplate(
        template="""
Answer the following question using only the provided context.
Use the conversation history only to resolve references (e.g. "it", "that")
- do not treat earlier answers as additional facts unless the document
context also supports them.

If the context does not contain enough information,
respond with "I don't know".

Conversation history:
{history}

Context:
{context}

Question:
{query}
""",
        input_variables=["history", "context", "query"]
    )

    final_prompt = prompt.format(
        history=history_text if history_text else "(none)",
        context=context,
        query=query
    )

    model = ChatGroq(
        model="qwen/qwen3.6-27b"
    )

    result = model.invoke(final_prompt)

    seen_citations = set()
    sources = []

    for doc in unique_docs[:5]:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page")

        # PyMuPDF pages are 0-indexed, so add 1 for display
        page_display = int(page) + 1 if page is not None else None

        citation = (source, page_display)
        if citation not in seen_citations:
            seen_citations.add(citation)
            sources.append({"file": source, "page": page_display})

    save_turn(user_id, document_id, query, result.content)

    return {"answer": result.content, "sources": sources}


def main():
    parser = argparse.ArgumentParser(description="Ask a question of one owned document.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--persist-dir", default="./chroma_db")
    args = parser.parse_args()

    result = run_query(args.user_id, args.document_id, args.query, args.persist_dir)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
