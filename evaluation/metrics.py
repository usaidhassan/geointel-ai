"""
Pure retrieval metrics - no I/O, no LLM calls, fully unit-testable.
Matches the definitions used in M4:
  Hit Rate: fraction of queries where the relevant chunk appears anywhere
            in the top-k results.
  MRR:      mean of 1/rank_of_first_relevant_result (0 if not found).
"""
from __future__ import annotations


def hit_rate(relevance_lists: list[list[bool]]) -> float:
    """relevance_lists[i] is a list of booleans (one per returned result,
    ranked best-first) indicating whether that result was the relevant one,
    for query i."""
    if not relevance_lists:
        return 0.0
    hits = sum(1 for rel in relevance_lists if any(rel))
    return hits / len(relevance_lists)


def mrr(relevance_lists: list[list[bool]]) -> float:
    if not relevance_lists:
        return 0.0
    total = 0.0
    for rel in relevance_lists:
        for rank, is_relevant in enumerate(rel, start=1):
            if is_relevant:
                total += 1.0 / rank
                break
    return total / len(relevance_lists)


def relevance_from_results(results: list[dict], expected_chunk_id: str, id_key: str = "chunk_id") -> list[bool]:
    """Turns a list of search result dicts into the boolean relevance list
    hit_rate/mrr expect, given the one chunk_id we know is the ground-truth
    source for this question."""
    return [r.get(id_key) == expected_chunk_id for r in results]
