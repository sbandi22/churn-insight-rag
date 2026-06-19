"""Build (or rebuild) the ChromaDB vector store from all knowledge sources.

Run:
    python -m ingestion.build_index

This loads papers, reports, and Telco row summaries, chunks them, embeds
them with the configured embedding model, and persists them to ChromaDB.
"""
from __future__ import annotations

import shutil
import sys

from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION_NAME, TELCO_CSV
from ingestion.chunking import chunk_documents
from ingestion.embeddings import get_embeddings, embedding_label
from ingestion.load_documents import load_all_documents, document_stats


def build_index(reset: bool = True) -> Chroma:
    """Build the Chroma vector store and return the client."""
    if reset and CHROMA_DIR.exists():
        print(f"Removing existing index at {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    if not TELCO_CSV.exists():
        print(
            "[warn] Telco CSV not found. Run 'python data/download_telco.py' "
            "first to include customer records. Continuing with papers/reports only."
        )

    print("Loading documents...")
    docs = load_all_documents()
    if not docs:
        raise SystemExit(
            "No documents found. Add files under data/papers, data/reports, "
            "or generate the Telco CSV."
        )

    stats = document_stats(docs)
    print(f"Loaded {len(docs)} documents: {stats}")

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print(f"Embedding with {embedding_label()} and writing to Chroma...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Index built: {len(chunks)} chunks persisted to {CHROMA_DIR}.")
    return vectorstore


def get_vectorstore() -> Chroma:
    """Open the existing persisted Chroma store (read path)."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


if __name__ == "__main__":
    try:
        build_index()
    except SystemExit as exc:
        print(exc)
        sys.exit(1)
