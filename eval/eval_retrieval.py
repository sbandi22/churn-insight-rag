"""
Retrieval evaluation script for the Churn Insight RAG system.

Runs 10 predefined questions through the retrieval pipeline and scores
the relevance of the top-5 retrieved chunks using keyword overlap.

Usage:
    python -m eval.eval_retrieval
    # or with verbose output:
    python -m eval.eval_retrieval --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.build_index import get_vectorstore  # noqa: E402
from retrieval.retriever import ChurnRetriever  # noqa: E402

# ── Evaluation questions with expected keyword signals ────────────────────────
EVAL_QUESTIONS: list[dict] = [
    {
        "question": "Why are enterprise customers churning?",
        "keywords": ["enterprise", "churn", "contract", "support", "cancel", "reason"],
    },
    {
        "question": "What churn rate is typical for SaaS companies?",
        "keywords": ["saas", "churn rate", "annual", "monthly", "benchmark", "percent"],
    },
    {
        "question": "Which customer segments are highest risk for churn?",
        "keywords": ["segment", "risk", "high", "month-to-month", "fiber", "senior"],
    },
    {
        "question": "How does contract type affect churn probability?",
        "keywords": ["contract", "month-to-month", "two-year", "one-year", "churn", "probability"],
    },
    {
        "question": "What does RFM analysis reveal about churn prediction?",
        "keywords": ["rfm", "recency", "frequency", "monetary", "churn", "prediction", "segmentation"],
    },
    {
        "question": "How can NLP on support tickets predict churn?",
        "keywords": ["nlp", "support", "ticket", "sentiment", "text", "predict", "churn"],
    },
    {
        "question": "What are the most effective retention strategies?",
        "keywords": ["retention", "strategy", "discount", "proactive", "win-back", "reduce", "churn"],
    },
    {
        "question": "How does tenure affect churn likelihood in the Telco dataset?",
        "keywords": ["tenure", "months", "churn", "long-term", "loyalty", "telco"],
    },
    {
        "question": "What is the average churn rate in the telecom industry?",
        "keywords": ["telecom", "churn rate", "industry", "average", "percent", "annual"],
    },
    {
        "question": "How do ensemble models improve churn prediction accuracy?",
        "keywords": ["ensemble", "random forest", "gradient boosting", "accuracy", "auc", "churn"],
    },
]


@dataclass
class EvalResult:
    question: str
    retrieved_chunks: int = 0
    keyword_hits: int = 0
    total_keywords: int = 0
    relevance_score: float = 0.0
    latency_ms: float = 0.0
    chunk_sources: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        """A result passes if at least 40% of keywords appear in retrieved chunks."""
        return self.relevance_score >= 0.4


def keyword_relevance_score(chunks: list, keywords: list[str]) -> tuple[int, int, float]:
    """Compute keyword overlap score between retrieved chunks and expected keywords."""
    combined_text = " ".join(
        (c.page_content if hasattr(c, "page_content") else str(c)).lower()
        for c in chunks
    )
    hits = sum(1 for kw in keywords if kw.lower() in combined_text)
    total = len(keywords)
    score = hits / total if total > 0 else 0.0
    return hits, total, score


def run_eval(verbose: bool = False) -> list[EvalResult]:
    """Run all evaluation questions and return results."""
    print("Loading vector store...")
    try:
        vs = get_vectorstore()
        retriever = ChurnRetriever(vectorstore=vs)
    except Exception as exc:
        print(f"ERROR: Could not load vector store: {exc}")
        print("Run 'python -m ingestion.build_index' first to build the index.")
        sys.exit(1)

    print(f"Running {len(EVAL_QUESTIONS)} evaluation questions...\n")
    results: list[EvalResult] = []

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        question = item["question"]
        keywords = item["keywords"]
        result = EvalResult(question=question, total_keywords=len(keywords))

        try:
            start = time.perf_counter()
            chunks = retriever.retrieve(question)
            elapsed_ms = (time.perf_counter() - start) * 1000

            hits, total, score = keyword_relevance_score(chunks, keywords)
            sources = [
                c.metadata.get("source", "unknown")
                for c in chunks
                if hasattr(c, "metadata")
            ]

            result.retrieved_chunks = len(chunks)
            result.keyword_hits = hits
            result.total_keywords = total
            result.relevance_score = score
            result.latency_ms = elapsed_ms
            result.chunk_sources = sources

        except Exception as exc:
            result.error = str(exc)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        score_pct = f"{result.relevance_score:.0%}"
        print(f"Q{i:02d} {status} [{score_pct}] ({result.latency_ms:.0f}ms) {question[:65]}")

        if verbose:
            print(f"     Keywords matched: {result.keyword_hits}/{result.total_keywords}")
            print(f"     Retrieved chunks: {result.retrieved_chunks}")
            print(f"     Sources: {list(dict.fromkeys(result.chunk_sources))[:3]}")
            if result.error:
                print(f"     ERROR: {result.error}")
            print()

        results.append(result)

    return results


def print_summary(results: list[EvalResult]) -> None:
    """Print aggregate evaluation summary."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    errored = sum(1 for r in results if r.error)
    avg_score = sum(r.relevance_score for r in results) / total if total else 0
    avg_latency = sum(r.latency_ms for r in results) / total if total else 0

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Questions evaluated : {total}")
    print(f"Passed (score >= 40%): {passed}/{total}  ({passed/total:.0%})")
    print(f"Errored             : {errored}")
    print(f"Avg relevance score : {avg_score:.2%}")
    print(f"Avg retrieval time  : {avg_latency:.0f} ms")
    print("=" * 60)

    if passed == total:
        print("All questions passed! Index quality looks good.")
    elif passed >= total * 0.7:
        print(f"Most questions passed ({passed}/{total}). Consider adding more data.")
    else:
        print(f"Only {passed}/{total} passed. Rebuild the index or add more documents.")


def save_results(results: list[EvalResult], output_path: Path) -> None:
    """Save evaluation results to a JSON file."""
    data = [
        {
            "question": r.question,
            "passed": r.passed,
            "relevance_score": round(r.relevance_score, 4),
            "keyword_hits": r.keyword_hits,
            "total_keywords": r.total_keywords,
            "retrieved_chunks": r.retrieved_chunks,
            "latency_ms": round(r.latency_ms, 1),
            "chunk_sources": r.chunk_sources,
            "error": r.error,
        }
        for r in results
    ]
    output_path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-question details")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Save results to JSON file (e.g., eval/results.json)",
    )
    args = parser.parse_args()

    results = run_eval(verbose=args.verbose)
    print_summary(results)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
