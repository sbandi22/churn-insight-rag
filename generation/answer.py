"""Generate grounded, cited answers from retrieved chunks.

Three providers are supported via LLM_PROVIDER:
  - "offline" (default): composes an extractive, fully-cited answer with no
    external API call, so the retrieval layer can be demoed with no key.
  - "openai": uses GPT-4o-mini.
  - "anthropic": uses Claude Haiku.

Every answer includes inline [n] citations that map to the retrieved chunks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
)
from retrieval.retriever import get_retriever, RetrievedChunk

SYSTEM_PROMPT = (
    "You are a customer-churn analyst. Answer the question USING ONLY the "
    "numbered context passages below. Cite every claim with its source number in "
    "square brackets, e.g. [1]. If the context does not contain the answer, say so "
    "plainly. Be concise and factual."
)


@dataclass
class AnswerResult:
    """An answer plus the chunks that ground it."""

    answer: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    provider: str = "offline"


def _format_context(chunks: List[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""
    blocks = []
    for c in chunks:
        blocks.append(f"[{c.rank}] ({c.title} -- {c.source})\n{c.text}")
    return "\n\n".join(blocks)


def _tokenize(text: str) -> List[str]:
    return ["".join(ch for ch in w if ch.isalnum()).lower() for w in text.split()]


def _split_sentences(text: str) -> List[str]:
    out, cur = [], []
    for ch in text:
        cur.append(ch)
        if ch in ".!?":
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def _offline_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    """Compose an extractive, cited answer without an LLM.

    We surface the most relevant sentences from the top chunks and attach their
    source numbers, so every sentence is traceable.
    """
    if not chunks:
        return (
            "I could not find anything relevant in the indexed documents to "
            "answer that question."
        )

    q_terms = {w for w in _tokenize(question) if len(w) > 3}
    lines = []
    for c in chunks[:3]:
        sentences = [s for s in _split_sentences(c.text) if s.strip()]
        sentences.sort(
            key=lambda s: len(q_terms & set(_tokenize(s))), reverse=True
        )
        best = sentences[0].strip() if sentences else c.text.strip()[:220]
        lines.append(f"- {best} [{c.rank}]")

    header = "Based on the indexed churn knowledge base:"
    return header + "\n" + "\n".join(lines)


def _llm_answer_openai(question: str, context: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}",
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def _llm_answer_anthropic(question: str, context: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}",
            }
        ],
    )
    return msg.content[0].text.strip()


def answer_question(question: str) -> AnswerResult:
    """Retrieve chunks and produce a grounded, cited answer."""
    chunks = get_retriever().retrieve(question)
    provider = LLM_PROVIDER.lower()

    try:
        if provider == "openai" and OPENAI_API_KEY:
            text = _llm_answer_openai(question, _format_context(chunks))
        elif provider == "anthropic" and ANTHROPIC_API_KEY:
            text = _llm_answer_anthropic(question, _format_context(chunks))
        else:
            provider = "offline"
            text = _offline_answer(question, chunks)
    except Exception as exc:  # pragma: no cover - graceful fallback
        provider = "offline (LLM error)"
        text = _offline_answer(question, chunks) + f"\n\n(note: LLM call failed: {exc})"

    return AnswerResult(answer=text, chunks=chunks, provider=provider)
