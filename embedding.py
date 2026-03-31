import re
import shutil
import argparse
import yaml
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

OPENAI_API_KEY = config.get("openai_key")
EXTRACTED_TEXT_FILE = "extracted_text.txt"
CHROMA_DB_PATH = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def text_to_documents(full_text):
    """Split text into Documents"""
    image_pattern = re.compile(
        r'\[\[IMAGE_DATA_START\]\]Image URL: (.*?), Image description: (.*?)\[\[IMAGE_DATA_END\]\]',
        re.DOTALL,
    )

    documents = []
    last_end = 0

    for match in image_pattern.finditer(full_text):
        text_before = full_text[last_end:match.start()].strip()
        if text_before:
            documents.append(Document(
                page_content=text_before,
                metadata={"type": "text"},
            ))

        img_path = match.group(1).strip()
        description = match.group(2).strip()
        if description:
            documents.append(Document(
                page_content=f"[FIGURE: {img_path}]: {description}",
                metadata={"type": "image_description", "image_path": img_path},
            ))

        last_end = match.end()

    remaining = full_text[last_end:].strip()
    if remaining:
        documents.append(Document(
            page_content=remaining,
            metadata={"type": "text"},
        ))

    return documents


def build_index():
    """Build ChromaDB vector store from extracted text."""
    chroma_path = CHROMA_DB_PATH
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

    # If vector store already exists, load it
    if Path(chroma_path).exists():
        print(f"Loading existing vector store from {chroma_path}")
        return Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings,
        )

    # Read extracted text
    with open(EXTRACTED_TEXT_FILE, "r", encoding="utf-8") as f:
        full_text = f.read()

    print(f"Read {len(full_text):,} characters from {EXTRACTED_TEXT_FILE}")

    # Convert to documents (separating image blocks from text)
    raw_docs = text_to_documents(full_text)
    print(f"Created {len(raw_docs)} raw documents ({sum(1 for d in raw_docs if d.metadata['type'] == 'image_description')} image blocks)")

    # Split text documents, keep image documents whole
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    split_docs = []
    for doc in raw_docs:
        if doc.metadata["type"] == "image_description":
            split_docs.append(doc)
        else:
            split_docs.extend(text_splitter.split_documents([doc]))

    print(f"After splitting: {len(split_docs)} chunks")

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=chroma_path,
    )

    print(f"Done. Vector store saved to {chroma_path}/")
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or rebuild the ChromaDB vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing vector store and rebuild from scratch.",
    )
    args = parser.parse_args()

    if args.rebuild and Path(CHROMA_DB_PATH).exists():
        print(f"Deleting existing vector store at {CHROMA_DB_PATH}/...")
        shutil.rmtree(CHROMA_DB_PATH)
        print("Deleted.")

    build_index()
