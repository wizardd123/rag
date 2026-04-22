import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "db"
DATA_PATH = "data/books"


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    return loader.load()


def split_text(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
    )
    return splitter.split_documents(documents)


def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )

    db.persist()
    print("✅ Database created!")


def main():
    docs = load_documents()
    chunks = split_text(docs)
    save_to_chroma(chunks)


if __name__ == "__main__":
    main()
