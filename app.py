import os

os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

query = "What is Infosys revenue?"

results = vectorstore.similarity_search(query, k=3)

for r in results:
    print(r.page_content)
    print("-" * 50)

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

