import os
import shutil as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



PDF_FOLDER = "Data/raw_pdfs"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=400
)

if os.path.exists("./chroma_db"):
    st.rmtree("./chroma_db")

all_chunks = []

for file in os.listdir(PDF_FOLDER):

    if not file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_FOLDER, file)

    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source_file"] = file

    all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Done. Total vectors stored:", vectorstore._collection.count())