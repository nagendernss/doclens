from doclens.fusion import rrf


def test_item_high_in_both_wins():
    # idx 2 is rank1 in list A and rank1 in list B → highest fused
    out = rrf([[2, 0, 1], [2, 1, 0]])
    assert out[0][0] == 2


def test_single_list_preserves_order():
    assert [i for i, _ in rrf([[5, 3, 9]])] == [5, 3, 9]


def test_dedup_and_tiebreak_idx_asc():
    out = rrf([[0, 1], [1, 0]])          # symmetric → equal scores
    assert [i for i, _ in out] == [0, 1] # idx asc tiebreak
    assert len(out) == 2


def test_k_const_monotonicity():
    # larger k_const compresses rank gaps → rank-1 advantage shrinks
    small = dict(rrf([[0, 1]], k_const=1))
    big   = dict(rrf([[0, 1]], k_const=1000))
    assert (small[0] - small[1]) > (big[0] - big[1])


def test_empty():
    assert rrf([]) == [] and rrf([[]]) == []
