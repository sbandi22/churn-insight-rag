"""Central configuration for the Churn Insight RAG system.

All paths and tunable parameters live here so the rest of the codebase can
import a single source of truth. Values can be overridden via environment
variables (loaded from a local .env file when present).
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional at import time
    pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
REPORTS_DIR = DATA_DIR / "reports"
TELCO_CSV = DATA_DIR / "telco_churn.csv"
CHROMA_DIR = PROJECT_ROOT / ".chroma"
COLLECTION_NAME = "churn_insight"


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
# "sentence-transformers" keeps everything offline (no API key required).
# Set EMBEDDING_BACKEND=openai to use OpenAI embeddings instead.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
ST_EMBEDDING_MODEL = os.getenv("ST_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
TOP_K = int(os.getenv("TOP_K", "5"))            # final chunks returned
FETCH_K = int(os.getenv("FETCH_K", "20"))       # candidates considered by MMR
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))  # 0=diversity, 1=relevance


# --------------------------------------------------------------------------- #
# Generation / LLM
# --------------------------------------------------------------------------- #
# "offline" composes an extractive, cited answer with no external API call.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "offline")  # offline | openai | anthropic
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

# How many Telco rows to summarize and index (keeps the demo fast).
TELCO_SAMPLE_ROWS = int(os.getenv("TELCO_SAMPLE_ROWS", "150"))
