"""
Core RAG loop: search -> build prompt -> call LLM. Deliberately framework-free
(matches M1's "build it by hand first" philosophy) so every part of the flow
is inspectable and debuggable under a tight deadline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import psycopg
import minsearch

from core.config import config
from core.embedder import BaseEmbedder
from core.llm_client import responses_create
from core.search import hybrid_search

SYSTEM_PROMPT = """You are GeoIntel AI, an assistant that answers questions about
remote sensing and precision agriculture using ONLY the provided CONTEXT, which
consists of excerpts from real research papers. If the CONTEXT does not contain
enough information to answer, say so explicitly instead of guessing. Always
mention which paper(s) (by title) your answer draws from.
"""

PROMPT_TEMPLATE = """QUESTION: {question}

CONTEXT:
{context}

Answer the question using only the CONTEXT above. Cite paper titles inline."""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[{c.get('title', c['doc_id'])} / {c.get('section', '')}]\n{c['text']}"
        )
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[dict]) -> str:
    return PROMPT_TEMPLATE.format(
        question=question,
        context=build_context(chunks),
    )


@dataclass
class RagResult:
    answer: str
    retrieved_chunk_ids: list[str]
    model: str
    prompt_tokens: int
    completion_tokens: int
    response_time_ms: int
    raw_chunks: list[dict] = field(default_factory=list)


def answer_question(
    question: str,
    conn: psycopg.Connection,
    kw_index: minsearch.Index,
    embedder: BaseEmbedder,
    num_results: int = 5,
    model: str | None = None,
) -> RagResult:

    start = time.time()

    chunks = hybrid_search(
        conn,
        kw_index,
        embedder,
        question,
        num_results=num_results,
    )

    prompt = build_prompt(question, chunks)

    response = responses_create(
        models=config.RAG_MODELS,
        instructions=SYSTEM_PROMPT,
        input=prompt,
    )

    elapsed_ms = int((time.time() - start) * 1000)

    usage = getattr(response, "usage", None)

    return RagResult(
        answer=response.output_text,
        retrieved_chunk_ids=[c["chunk_id"] for c in chunks],
        model=getattr(response, "used_model", "unknown"),
        prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
        completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        response_time_ms=elapsed_ms,
        raw_chunks=chunks,
    )