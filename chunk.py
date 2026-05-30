import os
import pickle

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

PDF_FOLDER = "data/raw_pdfs"
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

    print(f"Processing {file}")

    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source_file"] = file

    all_chunks.extend(chunks)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(all_chunks, f)

print(f"Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")