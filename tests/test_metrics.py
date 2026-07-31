from evaluation.metrics import hit_rate, mrr, relevance_from_results


def test_hit_rate_and_mrr_known_values():
    rel_lists = [
        [True, False, False],   # found at rank 1
        [False, False, True],   # found at rank 3
        [False, False, False],  # not found
    ]
    assert abs(hit_rate(rel_lists) - 2 / 3) < 1e-9
    assert abs(mrr(rel_lists) - (1 + 1 / 3) / 3) < 1e-9


def test_hit_rate_and_mrr_empty_input():
    assert hit_rate([]) == 0.0
    assert mrr([]) == 0.0


def test_hit_rate_perfect_score():
    rel_lists = [[True], [True, False]]
    assert hit_rate(rel_lists) == 1.0
    assert mrr(rel_lists) == 1.0


def test_relevance_from_results():
    results = [{"chunk_id": "x_001"}, {"chunk_id": "x_002"}, {"chunk_id": "a_000"}]
    assert relevance_from_results(results, "a_000") == [False, False, True]
    assert relevance_from_results(results, "not_present") == [False, False, False]
