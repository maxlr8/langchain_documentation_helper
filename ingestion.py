# pylint: disable-all

import os
import pinecone

from dotenv import load_dotenv

from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

env_path = ".backend/.env"
load_dotenv(env_path)

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

INDEX_NAME = os.environ["INDEX_NAME"]

def ingest_docs():

    loader = ReadTheDocsLoader("python.langchain.com/en/latest/index.html")
    raw_documents = loader.load()

    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = OpenAIEmbeddings()

    print(f"Going to add {len(documents)} to Pinecone")

    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to Vectore-store done***")

if __name__ == "__main__":
    ingest_docs()