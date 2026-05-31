import os
import pickle

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FOLDER = "Data/raw_pdfs"  
OUTPUT_FILE = "chunks.pkl"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

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

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(all_chunks, f)

print("Done!")