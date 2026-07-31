"""
Integration tests against a real Postgres+pgvector instance. Skipped
automatically if no DB is reachable (e.g. running `pytest` without first
starting docker-compose) rather than failing the whole suite.

Run with a DB available:
    docker-compose up -d db
    python -m monitoring.db_init
    pytest tests/test_search_integration.py
"""
import psycopg
import pytest
from pgvector.psycopg import register_vector

from core.config import config
from core.embedder import HashEmbedder
from core.search import build_keyword_index, hybrid_search, keyword_search, load_documents, vector_search

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def db_conn():
    try:
        conn = psycopg.connect(config.pg_conninfo(), connect_timeout=3)
    except Exception:
        pytest.skip("Postgres not reachable - start it with `docker-compose up -d db` to run this test")
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def seeded_docs(db_conn):
    register_vector(db_conn)
    embedder = HashEmbedder(dim=config.EMBEDDING_DIM)
    docs = [
        {"chunk_id": "it_a", "doc_id": "a", "section": "abstract", "title": "Crop Classification with Satellite and UAV",
         "text": "Deep learning for crop classification using satellite imagery and UAV multispectral data.", "authors": "Doe"},
        {"chunk_id": "it_b", "doc_id": "b", "section": "abstract", "title": "Roman Pottery Archaeology",
         "text": "Ancient Roman pottery excavation techniques and archaeological dating methods.", "authors": "Rossi"},
    ]
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE chunk_id LIKE 'it_%'")
        for d in docs:
            vec = embedder.encode([d["text"]])[0]
            cur.execute(
                "INSERT INTO documents (chunk_id, doc_id, section, text, title, authors, embedding) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (chunk_id) DO UPDATE SET text=EXCLUDED.text",
                (d["chunk_id"], d["doc_id"], d["section"], d["text"], d["title"], d["authors"], vec),
            )
    db_conn.commit()
    yield docs, embedder
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE chunk_id LIKE 'it_%'")
    db_conn.commit()


def test_keyword_search_finds_relevant_doc(db_conn, seeded_docs):
    all_docs = load_documents(db_conn)
    kw_index = build_keyword_index(all_docs)
    # Note: unlike vector search, keyword search naturally excludes documents
    # with zero term overlap (it_b, about Roman pottery, won't appear at all
    # for this query - that's correct TF-IDF behavior, not a bug), so this
    # test only checks that the relevant doc is found and ranks near the top.
    results = keyword_search(kw_index, "satellite crop classification", num_results=5)
    result_ids = [r["chunk_id"] for r in results]
    assert "it_a" in result_ids
    assert result_ids.index("it_a") <= 2


def test_vector_search_ranks_relevant_doc_above_unrelated(db_conn, seeded_docs):
    _, embedder = seeded_docs
    qvec = embedder.encode(["satellite imagery for crop monitoring"])[0]
    # num_results is large on purpose: this table is shared across the test
    # suite (and any data you've ingested locally), so we can't assume our
    # two seeded docs land in a small top-k - we just need both present to
    # compare their relative order.
    results = vector_search(db_conn, qvec, num_results=1000)
    ids_in_order = [r["chunk_id"] for r in results if r["chunk_id"] in ("it_a", "it_b")]
    assert ids_in_order.index("it_a") < ids_in_order.index("it_b")


def test_hybrid_search_returns_fused_results(db_conn, seeded_docs):
    all_docs = load_documents(db_conn)
    kw_index = build_keyword_index(all_docs)
    _, embedder = seeded_docs
    results = hybrid_search(db_conn, kw_index, embedder, "satellite crop classification", num_results=5)
    assert any(r["chunk_id"] == "it_a" for r in results)
    assert "rrf_score" in results[0]
