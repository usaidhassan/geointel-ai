"""
The "built-in judge" from M5: after every answered question, asynchronously
ask an LLM to rate the answer's relevance and store it in `feedback` with
source='judge'. This is what lets the monitoring dashboard show a relevance
trend even before any real user feedback comes in.

Kept separate from evaluation/evaluate_rag.py's judge_answer because this one
runs online (one question at a time, fire-and-forget from the API), while
the evaluation one runs offline in a batch. Same rating scheme, different
call site - duplicating the ~10 lines was simpler than sharing a dependency
between the api/ and evaluation/ packages under this deadline.
"""
from __future__ import annotations

from enum import Enum

import psycopg
from pydantic import BaseModel

from core.config import config
from core.llm_client import responses_parse
from monitoring.db_save import save_judge_feedback

JUDGE_PROMPT = """You are monitoring the quality of an AI assistant's answers about
remote sensing / precision agriculture research.

QUESTION: {question}
ANSWER: {answer}

Rate the answer's relevance as RELEVANT, PARTLY_RELEVANT, or NON_RELEVANT,
with a one-sentence explanation.
"""


class Relevance(str, Enum):
    RELEVANT = "RELEVANT"
    PARTLY_RELEVANT = "PARTLY_RELEVANT"
    NON_RELEVANT = "NON_RELEVANT"


class JudgeVerdict(BaseModel):
    relevance: Relevance
    explanation: str


def judge_and_store(
    conversation_id: int,
    question: str,
    answer: str,
    model: str | None = None,
):
    """Runs in a FastAPI BackgroundTask after the response is sent."""

    try:
        prompt = JUDGE_PROMPT.format(
            question=question,
            answer=answer,
        )

        response = responses_parse(
            models=config.JUDGE_MODELS,
            input=prompt,
            text_format=JudgeVerdict,
        )

        verdict = response.output_parsed

        with psycopg.connect(config.pg_conninfo()) as conn:
            save_judge_feedback(
                conn,
                conversation_id,
                verdict.relevance.value,
                verdict.explanation,
            )

    except Exception as e:
        print(f"[judge] failed for conversation {conversation_id}: {e}")