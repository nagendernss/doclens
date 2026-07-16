import pytest

from doclens.hybrid import HybridIndex
from doclens.types import Chunk


def _mk():
    idx = HybridIndex()
    chunks = [Chunk(f"d-{i:04d}", "d", 1, i, t) for i, t in enumerate([
        "mitochondria powerhouse of the cell",     # 0 lexical+dense for mito
        "cellular energy organelles respiration",  # 1
        "the quick brown fox jumps",               # 2 unrelated
    ])]
    vecs = [[1, 0, 0], [0.6, 0.4, 0], [0, 0, 1]]
    idx.add(chunks, vecs)
    return idx


def test_dense_mode_matches_vectorindex_order():
    idx = _mk()
    got = [r.chunk.seq for r in idx.retrieve([1, 0, 0], "x", mode="dense", pool=3)]
    assert got == [r.chunk.seq for r in idx.dense.search([1, 0, 0], k=3)]


def test_components_populated_and_dense_score_always_present():
    idx = _mk()
    out = idx.retrieve([1, 0, 0], "mitochondria", mode="hybrid", pool=3)
    for r in out:
        assert "dense_score" in r.components          # every candidate
        assert "rrf_score" in r.components            # hybrid only
    # mitochondria: rank-1 in both dense and bm25 → fused first
    assert out[0].chunk.seq == 0


def test_bm25_rank_none_when_no_lexical_hit():
    idx = _mk()
    out = idx.retrieve([0, 0, 1], "fox", mode="hybrid", pool=3)
    top = next(r for r in out if r.chunk.seq == 2)
    assert top.components["bm25_rank"] is not None
    # a chunk with no query-term overlap has bm25_rank None
    assert any(r.components["bm25_rank"] is None for r in out)


def test_pool_caps_length_and_len():
    idx = _mk()
    assert len(idx) == 3
    assert len(idx.retrieve([1, 0, 0], "cell", mode="hybrid", pool=2)) == 2


def test_unknown_mode_raises():
    # A typo'd mode must fail loudly, not silently run hybrid without rrf_score.
    idx = _mk()
    with pytest.raises(ValueError):
        idx.retrieve([1, 0, 0], "cell", mode="bogus", pool=3)


def test_negative_pool_returns_empty():
    idx = _mk()
    assert idx.retrieve([1, 0, 0], "cell", mode="hybrid", pool=-1) == []
