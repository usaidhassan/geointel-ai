from ingestion.fetch_papers import build_query, parse_atom_feed

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id>
    <published>2024-01-15T00:00:00Z</published>
    <title>Deep Learning for Crop Classification Using Sentinel-2 and UAV Imagery</title>
    <summary>We propose a CNN-based method for crop classification.</summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
    <link href="http://arxiv.org/abs/2401.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.01234v2" rel="related" type="application/pdf"/>
  </entry>
</feed>"""


def test_parse_atom_feed_extracts_expected_fields():
    papers = parse_atom_feed(SAMPLE_ATOM_XML)
    assert len(papers) == 1
    p = papers[0]
    assert p.arxiv_id == "2401.01234v2"
    assert "Crop Classification" in p.title
    assert p.authors == ["Jane Doe", "John Smith"]
    assert p.pdf_url == "http://arxiv.org/pdf/2401.01234v2"


def test_parse_atom_feed_empty_feed():
    empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert parse_atom_feed(empty) == []


def test_build_query_combines_categories_and_keywords():
    q = build_query(["eess.IV", "cs.CV"], ["remote sensing", "precision agriculture"])
    assert "cat:eess.IV" in q
    assert "cat:cs.CV" in q
    assert 'abs:"remote sensing"' in q
    assert 'abs:"precision agriculture"' in q
