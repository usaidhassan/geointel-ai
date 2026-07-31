"""
Creates the full schema: one Postgres instance serves as both the document
+ vector store (for retrieval) and the monitoring store (conversations,
feedback) - see the architecture rationale in the project handbook for why
this is one database instead of three services.

Run with: python -m monitoring.db_init
"""
import psycopg

from core.config import config

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

-- Retrieval corpus: one row per chunk
CREATE TABLE IF NOT EXISTS documents (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    section      TEXT,
    text         TEXT NOT NULL,
    title        TEXT,
    authors      TEXT,
    source_url   TEXT,
    embedding    vector({config.EMBEDDING_DIM}),
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents (doc_id);

-- HNSW index for fast approximate nearest-neighbor search.
-- Built AFTER ingestion in practice (see ingestion/chunk_and_ingest.py),
-- but declared here too so a fresh schema is self-consistent.
CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents USING hnsw (embedding vector_cosine_ops);

-- Monitoring: one row per question answered by the app
CREATE TABLE IF NOT EXISTS conversations (
    id                  BIGSERIAL PRIMARY KEY,
    question            TEXT NOT NULL,
    rewritten_query     TEXT,
    answer              TEXT NOT NULL,
    model               TEXT NOT NULL,
    retrieved_chunk_ids TEXT[],
    used_agent          BOOLEAN DEFAULT FALSE,
    tool_calls          INT DEFAULT 0,
    prompt_tokens       INT,
    completion_tokens   INT,
    cost_usd            NUMERIC(10, 6),
    response_time_ms    INT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- Feedback: one row per feedback event, from a real user OR the built-in judge
CREATE TABLE IF NOT EXISTS feedback (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT REFERENCES conversations(id) ON DELETE CASCADE,
    source          TEXT NOT NULL CHECK (source IN ('user', 'judge')),
    rating          INT,                 -- user: +1 / -1 thumbs
    relevance       TEXT,                -- judge: RELEVANT / PARTLY_RELEVANT / NON_RELEVANT
    explanation     TEXT,                -- judge's reasoning, if any
    created_at      TIMESTAMPTZ DEFAULT now()
);
"""


def init_db():
    with psycopg.connect(config.pg_conninfo()) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
    print("Schema created (documents, conversations, feedback).")


if __name__ == "__main__":
    init_db()
