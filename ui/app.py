"""
Streamlit Chat UI for the Churn Insight RAG system.

Run:
    streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make sure the project root is importable when launched from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHROMA_DIR, COLLECTION_NAME  # noqa: E402
from generation.answer import ChurnAnswerEngine  # noqa: E402
from ingestion.build_index import build_index, get_vectorstore  # noqa: E402

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Insight RAG",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine" not in st.session_state:
    st.session_state.engine = None
if "index_stats" not in st.session_state:
    st.session_state.index_stats = None

# ── Constants ────────────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "Why are enterprise customers churning?",
    "What churn rate is typical for SaaS companies?",
    "Which customer segments are highest risk?",
    "How does contract type affect churn probability?",
    "What does RFM analysis reveal about churn prediction?",
    "How can NLP on support tickets predict churn?",
    "What are the most effective retention strategies for telecom companies?",
    "How does tenure affect churn likelihood in the Telco dataset?",
    "What is the average churn rate in the telecom industry?",
    "How do ensemble models improve churn prediction accuracy?",
]

SOURCE_ICONS = {
    "paper": "📄",
    "report": "📊",
    "telco_customer": "👤",
}


# ── Cached resource loader ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading / building index…")
def load_engine() -> tuple[ChurnAnswerEngine, dict]:
    """Load or build the vector index and return the answer engine + stats."""
    if not CHROMA_DIR.exists():
        vs = build_index(reset=False)
    else:
        vs = get_vectorstore()

    try:
        col = vs._client.get_collection(COLLECTION_NAME)
        total_chunks = col.count()
    except Exception:
        total_chunks = "unknown"

    source_types: dict[str, int] = {}
    try:
        results = col.get(include=["metadatas"])
        for meta in results["metadatas"]:
            src = meta.get("source_type", "unknown")
            source_types[src] = source_types.get(src, 0) + 1
    except Exception:
        pass

    stats = {"total_chunks": total_chunks, "by_source": source_types}
    engine = ChurnAnswerEngine(vectorstore=vs)
    return engine, stats


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar(stats: dict | None) -> None:
    with st.sidebar:
        st.title("📉 Churn Insight RAG")
        st.markdown(
            "A RAG system for customer churn intelligence, grounded in "
            "Telco data, research papers, and industry reports."
        )
        st.markdown("---")

        st.header("📚 Indexed Documents")
        if stats is None:
            st.info("Loading index…")
        else:
            total = stats.get("total_chunks", "?")
            st.metric("Total Chunks", total)
            by_src = stats.get("by_source", {})
            if by_src:
                st.subheader("By Source Type")
                for src, count in sorted(by_src.items()):
                    icon = SOURCE_ICONS.get(src, "📁")
                    label = src.replace("_", " ").title()
                    st.markdown(f"{icon} **{label}**: {count} chunks")
            else:
                st.info("No per-source metadata available.")

        st.markdown("---")
        st.header("💡 Example Questions")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"btn_{hash(q)}", use_container_width=True):
                st.session_state.pending_question = q

        st.markdown("---")
        st.header("⚙️ Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rebuild Index", use_container_width=True):
                load_engine.clear()
                st.session_state.engine = None
                st.session_state.index_stats = None
                st.success("Index cleared — rebuilds on next query.")
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")
        st.caption(
            "Powered by **LangChain** · **ChromaDB** · **Sentence-Transformers**  \n"
            "Offline-capable (no API key required for retrieval)"
        )


# ── Message renderer ─────────────────────────────────────────────────────────
def render_sources_expander(sources: list[dict]) -> None:
    with st.expander(f"📌 {len(sources)} cited source(s)", expanded=False):
        for i, src in enumerate(sources, 1):
            doc_name = src.get("source", "Unknown document")
            chunk_id = src.get("chunk_id", "?")
            source_type = src.get("source_type", "")
            snippet = src.get("snippet", "")
            icon = SOURCE_ICONS.get(source_type, "📁")
            st.markdown(f"**{i}. {icon} {doc_name}** · chunk `{chunk_id}`")
            if snippet:
                display = snippet[:350] + ("…" if len(snippet) > 350 else "")
                st.markdown(f"> {display}")
            if i < len(sources):
                st.markdown("---")


def render_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            render_sources_expander(sources)


# ── Input handler ────────────────────────────────────────────────────────────
def handle_user_input(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    render_message("user", question)

    engine: ChurnAnswerEngine = st.session_state.engine

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer…"):
            try:
                result = engine.answer(question)
            except Exception as exc:
                error_msg = f"⚠️ Error generating answer: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
                return

        answer_text = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])
        st.markdown(answer_text)
        if sources:
            render_sources_expander(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    engine, stats = load_engine()
    st.session_state.engine = engine
    st.session_state.index_stats = stats

    render_sidebar(stats)

    st.title("💬 Churn Insight Assistant")
    st.markdown(
        "Ask any question about **customer churn** — powered by a RAG pipeline "
        "over Telco customer records, research papers, and industry reports."
    )
    st.markdown("---")

    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"], msg.get("sources"))

    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
        handle_user_input(question)
        st.rerun()

    if prompt := st.chat_input("Ask about customer churn…"):
        handle_user_input(prompt)


if __name__ == "__main__":
    main()
