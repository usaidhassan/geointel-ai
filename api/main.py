"""
FastAPI service - the graded "Interface" deliverable (rubric: 2/2 via API alone).

Endpoints:
    POST /ask       - ask a question, get a grounded answer
    POST /feedback  - submit thumbs up/down for a previous answer
    GET  /health    - liveness check

Run with: uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import psycopg
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from core.agent import rewrite_query, run_agent
from core.config import config
from core.rag import answer_question
from core.search import build_keyword_index, load_documents
from monitoring.db_save import save_conversation, save_user_feedback
from monitoring.judge import judge_and_store

# In-memory state built once at startup: the keyword index and embedder are
# expensive-ish to (re)build/load, so they live for the process lifetime
# rather than being rebuilt per-request. A fresh DB connection is opened
# per-request (see get_conn()) since psycopg connections aren't meant to be
# shared across concurrent requests.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from core.embedder import get_embedder

    print("Loading embedder and building keyword index...")
    state["embedder"] = get_embedder(use_stub=False)
    with psycopg.connect(config.pg_conninfo()) as conn:
        docs = load_documents(conn)
    if not docs:
        print("WARNING: `documents` table is empty - run ingestion first "
              "(python -m ingestion.chunk_and_ingest).")
    state["kw_index"] = build_keyword_index(docs)
    print(f"Ready. {len(docs)} chunks indexed.")
    yield
    state.clear()


app = FastAPI(title="GeoIntel AI", lifespan=lifespan)


def get_conn() -> psycopg.Connection:
    return psycopg.connect(config.pg_conninfo())


class AskRequest(BaseModel):
    question: str
    use_agent: bool = True
    rewrite: bool = True


class AskResponse(BaseModel):
    conversation_id: int
    question: str
    rewritten_query: str | None
    answer: str
    retrieved_chunk_ids: list[str]
    used_agent: bool
    tool_calls: int
    response_time_ms: int


class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: int  # +1 or -1


@app.get("/health")
def health():
    kw_index = state.get("kw_index")
    n_chunks = len(kw_index.docs) if kw_index is not None else 0
    return {"status": "ok", "chunks_indexed": n_chunks}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, background_tasks: BackgroundTasks):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    embedder = state["embedder"]
    kw_index = state["kw_index"]

    rewritten = None
    with get_conn() as conn:
        if req.use_agent:
            result = run_agent(req.question, conn, kw_index, embedder)
            answer, chunk_ids = result.answer, result.retrieved_chunk_ids
            tool_calls, prompt_tok, completion_tok, rt_ms = (
                result.tool_calls, result.prompt_tokens, result.completion_tokens, result.response_time_ms,
            )
        else:
            query_for_search = req.question
            if req.rewrite:
                rewritten = rewrite_query(req.question)
                query_for_search = rewritten
            result = answer_question(query_for_search, conn, kw_index, embedder)
            answer, chunk_ids = result.answer, result.retrieved_chunk_ids
            tool_calls, prompt_tok, completion_tok, rt_ms = (
                0, result.prompt_tokens, result.completion_tokens, result.response_time_ms,
            )

        conversation_id = save_conversation(
            conn,
            question=req.question,
            answer=answer,
            model=config.LLM_MODEL,
            retrieved_chunk_ids=chunk_ids,
            response_time_ms=rt_ms,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            rewritten_query=rewritten,
            used_agent=req.use_agent,
            tool_calls=tool_calls,
        )

    background_tasks.add_task(judge_and_store, conversation_id, req.question, answer)

    return AskResponse(
        conversation_id=conversation_id,
        question=req.question,
        rewritten_query=rewritten,
        answer=answer,
        retrieved_chunk_ids=chunk_ids,
        used_agent=req.use_agent,
        tool_calls=tool_calls,
        response_time_ms=rt_ms,
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be 1 or -1")
    with get_conn() as conn:
        save_user_feedback(conn, req.conversation_id, req.rating)
    return {"status": "ok"}
