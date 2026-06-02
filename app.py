import os
import pickle
import numpy as np
os.environ["HF_HUB_OFFLINE"] = "1"
from langchain_groq import ChatGroq
from dotenv import load_dotenv
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

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

corpus = [
    chunk.page_content.split()
    for chunk in chunks
]

bm25 = BM25Okapi(corpus)

query = input("Ask a question: ")

semantic_docs = retriever.invoke(query)

print(f"\nSemantic Results: {len(semantic_docs)}")

query_tokens = query.split()

scores = bm25.get_scores(query_tokens)

top_n = np.argsort(scores)[::-1][:5]

bm25_docs = [
    chunks[i]
    for i in top_n
]

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

context = "\n\n".join(
    doc.page_content
    for doc in unique_docs[:10]
)

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

for doc in unique_docs:

    source = doc.metadata.get("source_file")
    page = doc.metadata.get("page")

    print(f"{source} | Page {page}")