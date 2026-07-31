"""
Generates ground-truth (question, chunk_id) pairs by asking the LLM to write
plausible user questions for each chunk (structured output, Pydantic model),
following the A -> Q* pattern from M4. Parallelized since this is one call
per chunk and the corpus can have 1,000+ chunks.

Usage:
    python -m evaluation.generate_ground_truth --n-per-chunk 3 --sample 300
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg
from pydantic import BaseModel
from tqdm import tqdm

from core.config import config
from core.llm_client import client


class QuestionSet(BaseModel):
    questions: list[str]


PROMPT_TEMPLATE = """You are generating evaluation data for a search system.
Given a chunk of text from a research paper, write {n} short, realistic
questions a curious reader (a GIS/remote sensing student or practitioner)
might ask that this chunk would answer. Questions should be answerable from
this chunk ALONE, phrased naturally, and not just a copy of a sentence.

CHUNK (from paper titled "{title}", section "{section}"):
{text}
"""


def generate_questions_for_chunk(chunk: dict, n: int, model: str) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(n=n, title=chunk.get("title", ""), section=chunk.get("section", ""), text=chunk["text"])
    response = client().responses.parse(
        model=model,
        input=prompt,
        text_format=QuestionSet,
    )
    return response.output_parsed.questions


def fetch_all_chunks(conn: psycopg.Connection, sample: int | None) -> list[dict]:
    with conn.cursor() as cur:
        query = "SELECT chunk_id, doc_id, section, title, text FROM documents"
        if sample:
            query += f" ORDER BY random() LIMIT {sample}"
        cur.execute(query)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-chunk", type=int, default=3)
    parser.add_argument("--sample", type=int, default=None, help="only generate for a random sample of N chunks (faster/cheaper)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="data/ground_truth.csv")
    args = parser.parse_args()

    with psycopg.connect(config.pg_conninfo()) as conn:
        chunks = fetch_all_chunks(conn, args.sample)

    print(f"Generating {args.n_per_chunk} questions each for {len(chunks)} chunks...")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_questions_for_chunk, chunk, args.n_per_chunk, config.LLM_MODEL): chunk
            for chunk in chunks
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            chunk = futures[future]
            try:
                questions = future.result()
                for q in questions:
                    rows.append({"question": q, "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"]})
            except Exception as e:
                print(f"  Failed for {chunk['chunk_id']}: {e}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "chunk_id", "doc_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} ground-truth question/chunk_id pairs to {args.out}")


if __name__ == "__main__":
    main()
