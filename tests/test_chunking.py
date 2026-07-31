from core.chunking import chunk_document


def test_chunk_document_splits_by_section():
    text = """Abstract
Short abstract text about crop classification and remote sensing.

Results
Our model achieves 92 percent overall accuracy on the held out test set.

Conclusion
Combining satellite and UAV data improves crop classification accuracy.
"""
    chunks = chunk_document("doc1", text)
    sections = {c.section for c in chunks}
    assert sections == {"abstract", "results", "conclusion"}
    assert all(c.doc_id == "doc1" for c in chunks)
    assert all(c.chunk_id.startswith("doc1_") for c in chunks)


def test_chunk_document_splits_long_sections_with_overlap():
    long_intro = "Precision agriculture relies on remote sensing data. " * 40
    text = f"Introduction\n{long_intro}"
    chunks = chunk_document("doc2", text)
    assert len(chunks) > 1
    assert all(c.section == "introduction" for c in chunks)
    assert all(len(c.text.split()) <= 220 for c in chunks)


def test_chunk_document_falls_back_to_body_when_no_headers_found():
    text = "This is just a plain report with no standard academic section headers at all."
    chunks = chunk_document("doc3", text)
    assert len(chunks) == 1
    assert chunks[0].section == "body"


def test_chunk_document_drops_near_empty_fragments():
    text = "Abstract\nReal content here about crop yield prediction models.\n\nReferences\nA."
    chunks = chunk_document("doc4", text)
    # "A." alone is far too short to be a useful chunk
    assert all(len(c.text.split()) >= 2 for c in chunks)
