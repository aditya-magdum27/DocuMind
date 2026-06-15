import os
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from rank_bm25 import BM25Okapi

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

all_data = vectorstore._collection.get(include=["metadatas"])

unique_files = list(set(
    meta["source_file"] for meta in all_data["metadatas"]
))

print("\nAvailable PDFs:\n")
for i, file in enumerate(unique_files, start=1):
    print(f"  [{i}] {file}")

selection = input("\nSelect a PDF number: ").strip()
selected_file = unique_files[int(selection) - 1]

print(f"\nQuerying: {selected_file}\n")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 10,
        "filter": {"source_file": selected_file}
    }
)


chroma_data = vectorstore._collection.get(
    where={"source_file": selected_file},
    include=["documents", "metadatas"]
)

corpus = [text.split() for text in chroma_data["documents"]]

bm25 = BM25Okapi(corpus)

query = input("Ask a question: ")

semantic_docs = retriever.invoke(query)

print(f"\nSemantic Results: {len(semantic_docs)}")

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

print(f"BM25 Results: {len(bm25_docs)}")

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

print(f"Unique Results: {len(unique_docs)}")

context = "\n\n".join(doc.page_content for doc in unique_docs[:10])

print("\nRetrieved Chunks:\n")
print(context)

prompt = PromptTemplate(
    template="""
Answer the following question using only the provided context.

If the context does not contain enough information,
respond with "I don't know".

Context:
{context}

Question:
{query}
""",
    input_variables=["context", "query"]
)

final_prompt = prompt.format(
    context=context,
    query=query
)

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

result = model.invoke(final_prompt)

print("\nAnswer:\n")
print(result.content)

print("\nSources Used:\n")

seen_citations = set()

for doc in unique_docs[:5]:

    source = doc.metadata.get("source_file", "Unknown")
    page = doc.metadata.get("page")

    # PyMuPDF pages are 0-indexed, so add 1 for display
    page_display = int(page) + 1 if page is not None else "?"

    citation = (source, page_display)

    if citation not in seen_citations:
        seen_citations.add(citation)
        print(f"  File: {source}  |  Page: {page_display}")