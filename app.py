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

query = "What is Infosys revenue?"

docs = vectorstore.similarity_search(query, k=10)
context_text = "\n\n".join([doc.page_content for doc in docs])

print(context_text)
prompt = PromptTemplate(
    template=""""
        Answer the following question from given context only if the context is not provode info or u don't know it just return "i don't know"

        context:
        {context}

        question:
        {query}
    """,
    input_variables=['query','context']
)

final_prompt = prompt.format(
    context=context_text,
    query=query
)

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

result = model.invoke(final_prompt)

print(result.content)