from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
import os

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks from chunks.pkl")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Done. Total vectors stored:", vectorstore._collection.count())