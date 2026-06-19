"""MMR-based retriever over the persisted Chroma store.

Maximal Marginal Relevance (MMR) balances relevance with diversity so the
top-k chunks are not all near-duplicates. This matters here because many
Telco row summaries are textually similar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from config import TOP_K, FETCH_K, MMR_LAMBDA
from ingestion.build_index import get_vectorstore


@dataclass
class RetrievedChunk:
    """A single retrieved passage with the metadata needed to cite it."""

    rank: int
    text: str
    source: str
    doc_type: str
    title: str
    chunk_id: str

    @property
    def citation(self) -> str:
        return f"[{self.rank}] {self.title} ({self.source}, chunk {self.chunk_id})"


class ChurnRetriever:
    """Thin wrapper around a Chroma MMR retriever."""

    def __init__(self, top_k: int = TOP_K, fetch_k: int = FETCH_K, lambda_mult: float = MMR_LAMBDA):
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self._vectorstore = None

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = get_vectorstore()
        return self._vectorstore

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the top-k MMR chunks for a query."""
        docs: List[Document] = self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.top_k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )
        results: List[RetrievedChunk] = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            results.append(
                RetrievedChunk(
                    rank=i,
                    text=d.page_content,
                    source=meta.get("source", "unknown"),
                    doc_type=meta.get("doc_type", "unknown"),
                    title=meta.get("title", meta.get("source", "Untitled")),
                    chunk_id=str(meta.get("chunk_id", meta.get("chunk_index", i))),
                )
            )
        return results


_default_retriever: ChurnRetriever | None = None


def get_retriever() -> ChurnRetriever:
    """Return a process-wide singleton retriever."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = ChurnRetriever()
    return _default_retriever
