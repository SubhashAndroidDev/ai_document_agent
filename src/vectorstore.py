from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.ingest import load_documents


PDF_PATH = "documents/company_policy.pdf"
CHROMA_PATH = "chroma_db"


def create_vectorstore():

    print("Loading PDF...")

    chunks = load_documents(PDF_PATH)

    print(f"Loaded {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("Vector database created!")

    return vectorstore


def get_vectorstore():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vectorstore