import os
os.environ["HF_HUB_OFFLINE"] = "1"
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 10}
)

query = "What is the amount involved in frauds by public sector banks in 2025-26?"

docs = retriever.invoke(query)

context = "\n\n".join(
    doc.page_content for doc in docs
)

print(f"Retrieved {len(docs)} chunks\n")
print(context)
print("\n" + "="*60 + "\n")

prompt = PromptTemplate(
    template="""
        Answer the following question using only the provided context.
        If the context does not contain enough information, respond with "I don't know".

        context:
        {context}

        question:
        {query}
    """,
    input_variables=['query', 'context']
)

final_prompt = prompt.format(
    context=context,
    query=query
)

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

result = model.invoke(final_prompt)

print("This is the answer form llm " + result.content)
print("\nSources Used:")

seen = set()

for doc in docs:
    source = doc.metadata.get("source_file")
    page = doc.metadata.get("page")

    key = (source, page)

    if key not in seen:
        seen.add(key)
        print(f"{source} | Page {page}")