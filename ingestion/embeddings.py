"""Embedding-model factory.

Defaults to a local sentence-transformers model (all-MiniLM-L6-v2) so the
system runs fully offline with no API key. Set EMBEDDING_BACKEND=openai to
use OpenAI embeddings instead.
"""
from __future__ import annotations

from config import (
    EMBEDDING_BACKEND,
    ST_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


def get_embeddings():
    """Return a LangChain-compatible embeddings object based on config.

    Raises a clear error if the OpenAI backend is selected without a key.
    """
    backend = EMBEDDING_BACKEND.lower()

    if backend == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "EMBEDDING_BACKEND=openai but OPENAI_API_KEY is not set. "
                "Set the key or use EMBEDDING_BACKEND=sentence-transformers."
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    # Default: offline sentence-transformers.
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=ST_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embedding_label() -> str:
    """Human-readable name of the active embedding model (for the UI)."""
    if EMBEDDING_BACKEND.lower() == "openai":
        return f"OpenAI: {OPENAI_EMBEDDING_MODEL}"
    return f"sentence-transformers: {ST_EMBEDDING_MODEL}"
