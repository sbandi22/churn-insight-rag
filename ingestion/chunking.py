"""Split documents into overlapping chunks for embedding.

Uses LangChain's RecursiveCharacterTextSplitter so semantic boundaries
(paragraphs, sentences) are preferred over arbitrary character cuts.
Each resulting chunk carries a stable chunk_id and its source metadata
so the generation layer can cite exactly which passage was used.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def get_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    """Return a configured recursive text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split a list of LangChain Documents into chunks with citation metadata.

    Each chunk inherits its parent's metadata and gains:
      - chunk_index: position within the source document
      - chunk_id: a human-readable identifier like "source.csv:3"
    """
    splitter = get_splitter()
    split_docs = splitter.split_documents(docs)

    # Track per-source counters to build stable chunk ids.
    counters: dict[str, int] = {}
    for doc in split_docs:
        source = doc.metadata.get("source", "unknown")
        idx = counters.get(source, 0)
        doc.metadata["chunk_index"] = idx
        doc.metadata["chunk_id"] = f"{source}:{idx}"
        counters[source] = idx + 1

    return split_docs
