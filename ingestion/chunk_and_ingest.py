"""
Semi-automated ingestion pipeline (rubric: Ingestion pipeline, 1 point as a
script; upgradeable to 2 points by wrapping this in Kestra/Airflow/Prefect -
see the handbook's "if you finish early" list).

Reads data/raw_pdfs/manifest.json (written by ingestion/fetch_papers.py),
chunks every PDF, embeds every chunk, and upserts into the `documents` table.

Usage:
    python -m ingestion.chunk_and_ingest
    python -m ingestion.chunk_and_ingest --stub-embedder   # offline/dev mode
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from tqdm import tqdm

from core.chunking import chunk_pdf
from core.config import config
from core.embedder import get_embedder


def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def ingest(manifest_path: str = "data/raw_pdfs/manifest.json", use_stub_embedder: bool = False, batch_size: int = 32):
    manifest = load_manifest(Path(manifest_path))
    embedder = get_embedder(use_stub=use_stub_embedder)
    print(f"Embedder: {type(embedder).__name__} (dim={embedder.dim})")

    all_chunks = []
    for paper in tqdm(manifest, desc="Chunking PDFs"):
        pdf_path = paper["local_path"]
        doc_id = paper["arxiv_id"]
        try:
            chunks = chunk_pdf(
                pdf_path,
                doc_id,
                extra_metadata={
                    "title": paper.get("title", ""),
                    "authors": ", ".join(paper.get("authors", [])),
                    "source_url": paper.get("pdf_url", ""),
                },
            )
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Skipping {doc_id}: {e}")

    print(f"Total chunks: {len(all_chunks)}")

    with psycopg.connect(config.pg_conninfo()) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding + inserting"):
                batch = all_chunks[i : i + batch_size]
               # Clean text before embedding and inserting
                clean_texts = [c.text.replace("\x00", "") for c in batch]
                vectors = embedder.encode(clean_texts)

                for chunk, clean_text, vec in zip(batch, clean_texts, vectors):
                    cur.execute(
                        """
                        INSERT INTO documents (chunk_id, doc_id, section, text, title, authors, source_url, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            text = EXCLUDED.text, embedding = EXCLUDED.embedding
                        """,
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.section,
                            clean_text,
                            chunk.metadata.get("title", ""),
                            chunk.metadata.get("authors", ""),
                            chunk.metadata.get("source_url", ""),
                            vec,
                        ),
                    )
        conn.commit()

    print(f"Ingested {len(all_chunks)} chunks from {len(manifest)} papers into Postgres.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/raw_pdfs/manifest.json")
    parser.add_argument("--stub-embedder", action="store_true", help="use offline HashEmbedder instead of sentence-transformers")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    ingest(args.manifest, use_stub_embedder=args.stub_embedder, batch_size=args.batch_size)
