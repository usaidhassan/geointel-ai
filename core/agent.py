"""
Lightweight agent layer, built by hand (no framework) following the M1
function-calling / agentic-loop pattern: give the model ONE tool
(search_papers), let it call the tool, inspect results, and either answer or
retry with a reformulated query (capped at MAX_ITERATIONS to avoid loops).

Two separate mechanisms live here on purpose, because they satisfy two
different things:
  - rewrite_query(): one explicit LLM call that cleans up the raw user
    question before the *first* search. This is the standalone
    "query rewriting" best practice.
  - run_agent(): the tool-calling loop itself, which can rewrite AGAIN and
    retry mid-conversation if the first search comes back empty/off-topic.
    This is the "agentic" part - recovery behavior, not just rewriting.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import psycopg
import minsearch

from core.config import config
from core.embedder import BaseEmbedder
from core.llm_client import  responses_create
from core.rag import SYSTEM_PROMPT, build_context
from core.search import hybrid_search

MAX_ITERATIONS = 4

SEARCH_TOOL = {
    "type": "function",
    "name": "search_papers",
    "description": (
        "Search the remote sensing / precision agriculture research paper "
        "corpus for chunks relevant to a query. Call this again with a "
        "reformulated query if the first results look irrelevant or empty."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, ideally specific keywords rather than a full sentence.",
            }
        },
        "required": ["query"],
    },
}

AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n\nYou have a search_papers tool. Call it to find relevant context before "
    "answering. If results look irrelevant, sparse, or off-topic, try again "
    "with a reformulated, more specific query (at most 2 retries) before "
    "answering with what you have."
)


def rewrite_query(raw_question: str, model: str | None = None) -> str:
    """One explicit LLM call that turns a raw user question into a cleaner
    search query. Kept separate from the agent loop so it's independently
    demonstrable/evaluable as its own 'query rewriting' technique."""
    model = model or config.LLM_MODEL
    response = responses_create(
        models=config.AGENT_MODELS,
        instructions=(
            "Rewrite the user's question into a short, keyword-focused search "
            "query for a research-paper search engine. Return ONLY the rewritten "
            "query, no explanation, no quotes."
        ),
        input=raw_question,
    )
    return response.output_text.strip()


@dataclass
class AgentResult:
    answer: str
    trajectory: list[dict] = field(default_factory=list)  # log of tool calls made
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    tool_calls: int = 0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    response_time_ms: int = 0


def run_agent(
    question: str,
    conn: psycopg.Connection,
    kw_index: minsearch.Index,
    embedder: BaseEmbedder,
    model: str | None = None,
) -> AgentResult:
    model = model or config.LLM_MODEL
    start = time.time()

    def do_search(query: str) -> list[dict]:
        return hybrid_search(conn, kw_index, embedder, query, num_results=5)

    trajectory: list[dict] = []
    all_retrieved_ids: list[str] = []
    tool_call_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # conversation state passed to the Responses API across turns
    input_items: list[dict] = [{"role": "user", "content": question}]

    for _iteration in range(MAX_ITERATIONS):
        response = responses_create(
            models=config.AGENT_MODELS,
            instructions=AGENT_SYSTEM_PROMPT,
            input=input_items,
            tools=[SEARCH_TOOL],
        )

        usage = getattr(response, "usage", None)
        if usage:
            total_prompt_tokens += getattr(usage, "input_tokens", 0)
            total_completion_tokens += getattr(usage, "output_tokens", 0)

        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            # Model produced a final answer - done.
            return AgentResult(
                answer=response.output_text,
                trajectory=trajectory,
                retrieved_chunk_ids=all_retrieved_ids,
                tool_calls=tool_call_count,
                model=model,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                response_time_ms=int((time.time() - start) * 1000),
            )

        # Echo the model's own output items back into the conversation,
        # then append the tool results, per the Responses API tool-use contract.
        input_items.extend(response.output)

        for call in function_calls:
            args = json.loads(call.arguments)
            query = args.get("query", question)
            results = do_search(query)
            tool_call_count += 1
            chunk_ids = [r["chunk_id"] for r in results]
            all_retrieved_ids.extend(chunk_ids)
            trajectory.append({"tool": "search_papers", "query": query, "n_results": len(results)})

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": build_context(results) if results else "No results found.",
                }
            )

    # Hit MAX_ITERATIONS without a final answer - force a plain answer with
    # whatever context we've gathered rather than looping forever.
    input_items.append(
        {"role": "user", "content": "Please answer now with the best information you have gathered so far."}
    )
    final = responses_create(models=config.AGENT_MODELS, instructions=AGENT_SYSTEM_PROMPT, input=input_items)
    return AgentResult(
        answer=final.output_text,
        trajectory=trajectory,
        retrieved_chunk_ids=all_retrieved_ids,
        tool_calls=tool_call_count,
        model=model,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        response_time_ms=int((time.time() - start) * 1000),
    )
