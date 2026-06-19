"""Load all knowledge sources into LangChain Documents.

Three source types are supported:
  1. Research-paper abstracts    (data/papers/*.md)
  2. Industry-report excerpts   (data/reports/*.md)
  3. Telco churn row summaries  (data/telco_churn.csv -> NL summaries)

Each Document carries metadata used for citations: source, doc_type, title.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from config import PAPERS_DIR, REPORTS_DIR, TELCO_CSV, TELCO_SAMPLE_ROWS
from ingestion.telco_summarizer import summarize_row


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith("# "):
            return line.lstrip("# ").strip()
    return fallback


def _load_markdown_dir(directory: Path, doc_type: str) -> List[Document]:
    docs: List[Document] = []
    if not directory.exists():
        return docs
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = _title_from_markdown(text, path.stem)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "doc_type": doc_type,
                    "title": title,
                },
            )
        )
    return docs


def load_papers() -> List[Document]:
    """Load research-paper abstracts."""
    return _load_markdown_dir(PAPERS_DIR, "research_paper")


def load_reports() -> List[Document]:
    """Load industry-report excerpts."""
    return _load_markdown_dir(REPORTS_DIR, "industry_report")


def load_telco_summaries(max_rows: int = TELCO_SAMPLE_ROWS) -> List[Document]:
    """Read the Telco CSV and turn each row into a natural-language Document.

    Uses the stdlib csv module so no pandas dependency is required at load time.
    """
    docs: List[Document] = []
    if not TELCO_CSV.exists():
        return docs
    with open(TELCO_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            summary = summarize_row(row)
            cid = row.get("customerID", str(i))
            docs.append(
                Document(
                    page_content=summary,
                    metadata={
                        "source": "telco_churn.csv",
                        "doc_type": "customer_record",
                        "title": f"Telco customer {cid}",
                        "customer_id": cid,
                        "churn": row.get("Churn", ""),
                    },
                )
            )
    return docs


def load_all_documents() -> List[Document]:
    """Load papers + reports + Telco summaries into one list."""
    return load_papers() + load_reports() + load_telco_summaries()


def document_stats(docs: List[Document]) -> dict:
    """Count documents per doc_type (for the UI sidebar)."""
    stats: dict[str, int] = {}
    for d in docs:
        t = d.metadata.get("doc_type", "unknown")
        stats[t] = stats.get(t, 0) + 1
    return stats
