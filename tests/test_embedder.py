import numpy as np

from core.embedder import HashEmbedder


def test_hash_embedder_deterministic():
    e = HashEmbedder(dim=128)
    v1 = e.encode(["satellite crop monitoring"])
    v2 = e.encode(["satellite crop monitoring"])
    assert np.allclose(v1, v2)


def test_hash_embedder_shape():
    e = HashEmbedder(dim=128)
    vecs = e.encode(["one", "two", "three"])
    assert vecs.shape == (3, 128)


def test_hash_embedder_related_text_more_similar_than_unrelated():
    e = HashEmbedder(dim=384)
    a, b, c = e.encode([
        "satellite imagery crop classification precision agriculture",
        "UAV drone imagery crop classification precision agriculture",
        "ancient roman pottery archaeological excavation",
    ])
    assert (a @ b) > (a @ c)
