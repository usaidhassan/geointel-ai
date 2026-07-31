"""
LLM-as-a-judge evaluation of full RAG answers (rubric: LLM evaluation = 2/2).
For each ground-truth question, runs the actual RAG pipeline to get an
answer, then asks a judge model to rate it RELEVANT / PARTLY_RELEVANT /
NON_RELEVANT with a short explanation (the A -> Q -> A' pattern from M4).

Usage:
    python -m evaluation.evaluate_rag --sample 50
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import psycopg
from pydantic import BaseModel
from tqdm import tqdm

from core.config import config
from core.embedder import get_embedder
from core.llm_client import client
from core.rag import answer_question
from core.search import build_keyword_index, load_documents

JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's answer to a
question about remote sensing / precision agriculture research.

QUESTION: {question}

GENERATED ANSWER: {answer}

Rate the answer's relevance to the question as one of:
RELEVANT - directly and fully answers the question
PARTLY_RELEVANT - addresses the question but is incomplete or partially off-topic
NON_RELEVANT - does not answer the question

Provide your rating and a one-sentence explanation.
"""


class Relevance(str, Enum):
    RELEVANT = "RELEVANT"
    PARTLY_RELEVANT = "PARTLY_RELEVANT"
    NON_RELEVANT = "NON_RELEVANT"


class JudgeVerdict(BaseModel):
    relevance: Relevance
    explanation: str


def judge_answer(question: str, answer: str, model: str) -> JudgeVerdict:
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    response = client().responses.parse(model=model, input=prompt, text_format=JudgeVerdict)
    return response.output_parsed


def evaluate_one(row: dict, kw_index, embedder, model: str) -> dict:
    # Each worker thread opens its OWN Postgres connection - a single
    # psycopg connection must not be shared for concurrent cursor use
    # across threads. kw_index is read-only in-memory search, safe to share.
    with psycopg.connect(config.pg_conninfo()) as conn:
        result = answer_question(row["question"], conn, kw_index, embedder, model=model)
    verdict = judge_answer(row["question"], result.answer, model)
    return {
        "question": row["question"],
        "answer": result.answer,
        "relevance": verdict.relevance.value,
        "explanation": verdict.explanation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="data/ground_truth.csv")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--stub-embedder", action="store_true")
    parser.add_argument("--out", default="data/rag_eval_results.csv")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.ground_truth) as f:
        rows = list(csv.DictReader(f))[: args.sample]

    embedder = get_embedder(use_stub=args.stub_embedder)

    with psycopg.connect(config.pg_conninfo()) as conn:
        docs = load_documents(conn)
        kw_index = build_keyword_index(docs)

        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(evaluate_one, row, kw_index, embedder, config.LLM_MODEL): row
                for row in rows
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"  Failed: {e}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "relevance", "explanation"])
        writer.writeheader()
        writer.writerows(results)

    counts = {}
    for r in results:
        counts[r["relevance"]] = counts.get(r["relevance"], 0) + 1
    print("\nRelevance breakdown:")
    for k, v in counts.items():
        print(f"  {k}: {v} ({100*v/len(results):.1f}%)")
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
