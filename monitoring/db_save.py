"""
Writes one row per answered question to `conversations`, and one row per
feedback event (user thumbs up/down, or the built-in judge) to `feedback`.
Cost is computed from a small hardcoded per-model price table - update
PRICING if you change LLM_MODEL to a provider/model not listed here.
"""
from __future__ import annotations

import psycopg


# USD per 1M tokens (input, output). Keep this updated if you switch models
# via the Gateway - it's only used for the monitoring dashboard's cost chart,
# never for billing itself (the Gateway bills you directly at list price).
PRICING = {
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "groq/llama-3.3-70b-versatile": (0.0, 0.0),  # adjust if not on a free tier
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price


def save_conversation(
    conn: psycopg.Connection,
    question: str,
    answer: str,
    model: str,
    retrieved_chunk_ids: list[str],
    response_time_ms: int,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    rewritten_query: str | None = None,
    used_agent: bool = False,
    tool_calls: int = 0,
) -> int:
    cost = estimate_cost(model, prompt_tokens, completion_tokens)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations
                (question, rewritten_query, answer, model, retrieved_chunk_ids,
                 used_agent, tool_calls, prompt_tokens, completion_tokens,
                 cost_usd, response_time_ms)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (
                question, rewritten_query, answer, model, retrieved_chunk_ids,
                used_agent, tool_calls, prompt_tokens, completion_tokens,
                cost, response_time_ms,
            ),
        )
        conversation_id = cur.fetchone()[0]
    conn.commit()
    return conversation_id


def save_user_feedback(conn: psycopg.Connection, conversation_id: int, rating: int):
    """rating: +1 (thumbs up) or -1 (thumbs down)"""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO feedback (conversation_id, source, rating) VALUES (%s, 'user', %s)",
            (conversation_id, rating),
        )
    conn.commit()


def save_judge_feedback(conn: psycopg.Connection, conversation_id: int, relevance: str, explanation: str):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO feedback (conversation_id, source, relevance, explanation) VALUES (%s, 'judge', %s, %s)",
            (conversation_id, relevance, explanation),
        )
    conn.commit()
