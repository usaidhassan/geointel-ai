"""
Three retrieval methods, evaluated head-to-head in evaluation/evaluate_search.py:

1. keyword_search   - minsearch TF-IDF index, held in memory, rebuilt from
                       Postgres at process startup (corpus is small enough
                       that this takes seconds, not minutes).
2. vector_search     - pgvector cosine similarity, queried directly in Postgres.
3. hybrid_search     - Reciprocal Rank Fusion (RRF) over the two rankings
                       above. This is also this project's "reranking"
                       best-practice: RRF is rank fusion, not a bag-of-words
                       union, which is exactly what M6 describes.
"""
from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector
import minsearch
import numpy as np

from core.embedder import BaseEmbedder

RRF_K = 60  # standard default from the literature and from M6


def load_documents(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id, doc_id, section, text, title, authors FROM documents"
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def build_keyword_index(docs: list[dict]) -> minsearch.Index:
    index = minsearch.Index(
        text_fields=["text", "title", "section"],
        keyword_fields=["doc_id", "chunk_id"],
    )
    index.fit(docs)
    return index


def keyword_search(index: minsearch.Index, query: str, num_results: int = 10) -> list[dict]:
    return index.search(
        query,
        boost_dict={"title": 1.5, "text": 1.0, "section": 0.5},
        num_results=num_results,
    )


def vector_search(
    conn: psycopg.Connection, query_vector: np.ndarray, num_results: int = 10
) -> list[dict]:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, doc_id, section, text, title, authors,
                   1 - (embedding <=> %s) AS similarity
            FROM documents
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_vector, query_vector, num_results),
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _rrf_merge(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """ranked_lists: each is a list of chunk_ids in rank order (best first).
    Returns [(chunk_id, rrf_score), ...] sorted best-first."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_search(
    conn: psycopg.Connection,
    kw_index: minsearch.Index,
    embedder: BaseEmbedder,
    query: str,
    num_results: int = 10,
    candidate_pool: int = 30,
) -> list[dict]:
    """RRF fusion of keyword + vector rankings, then re-fetch full rows for
    the fused top-N in the fused order."""
    kw_results = keyword_search(kw_index, query, num_results=candidate_pool)
    query_vec = embedder.encode([query])[0]
    vec_results = vector_search(conn, query_vec, num_results=candidate_pool)

    kw_ranked_ids = [d["chunk_id"] for d in kw_results]
    vec_ranked_ids = [d["chunk_id"] for d in vec_results]
    fused = _rrf_merge([kw_ranked_ids, vec_ranked_ids])[:num_results]

    by_id = {d["chunk_id"]: d for d in kw_results}
    by_id.update({d["chunk_id"]: d for d in vec_results})
    return [by_id[chunk_id] | {"rrf_score": score} for chunk_id, score in fused if chunk_id in by_id]
