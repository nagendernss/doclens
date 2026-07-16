import math

import pytest

from doclens.index import VectorIndex
from doclens.types import Chunk


def C(i):
    return Chunk(chunk_id=f"c{i}", doc_id="d", page=1, seq=i, text=f"t{i}")


def test_cosine_ranking_hand_computed():
    idx = VectorIndex()
    idx.add([C(0), C(1), C(2)], [[1, 0], [1, 1], [0, 1]])
    out = idx.search([1, 0], k=2)
    assert [r.chunk.chunk_id for r in out] == ["c0", "c1"]
    assert math.isclose(out[0].score, 1.0, abs_tol=1e-9)
    assert math.isclose(out[1].score, 1 / math.sqrt(2), abs_tol=1e-9)


def test_mismatch_raises():
    idx = VectorIndex()
    with pytest.raises(ValueError):
        idx.add([C(0)], [[1, 0], [0, 1]])
    idx.add([C(0)], [[1, 0]])
    with pytest.raises(ValueError):
        idx.add([C(1)], [[1, 0, 0]])


def test_zero_vector_safe():
    idx = VectorIndex()
    idx.add([C(0), C(1)], [[0, 0], [1, 0]])
    out = idx.search([1, 0], k=5)
    assert out[0].chunk.chunk_id == "c1"
    assert all(not math.isnan(r.score) for r in out)


def test_len():
    idx = VectorIndex()
    assert len(idx) == 0
    idx.add([C(0)], [[1.0]])
    assert len(idx) == 1


def test_tie_stability():
    """Verify stable sort preserves insertion order for tied cosine scores."""
    idx = VectorIndex()
    # Add 6 chunks with alternating vectors: [1,0], [0,1], [1,0], [0,1], [1,0], [0,1]
    # All [1,0] and [0,1] will have score 1.0 when queried with [1,0]
    idx.add(
        [C(0), C(1), C(2), C(3), C(4), C(5)],
        [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
    )
    out = idx.search([1, 0], k=3)
    # All three results tie at score 1.0; insertion order should be c0, c2, c4
    assert [r.chunk.chunk_id for r in out] == ["c0", "c2", "c4"]
    assert all(math.isclose(r.score, 1.0, abs_tol=1e-9) for r in out)


def test_rank_all_returns_all_sorted_with_index():
    from doclens.index import VectorIndex
    from doclens.types import Chunk
    idx = VectorIndex()
    ch = [Chunk(f"d-{i}", "d", 1, i, t) for i, t in enumerate(["a","b","c"])]
    idx.add(ch, [[1,0],[0,1],[1,1]])
    ranked = idx.rank_all([1,0])
    assert len(ranked) == 3
    assert ranked[0][0] == 0                       # exact match cosine highest
    assert [i for i,_ in ranked] == sorted(range(3), key=lambda i: (-dict(ranked)[i], i))


def test_rank_all_empty():
    from doclens.index import VectorIndex
    assert VectorIndex().rank_all([1,0]) == []


def test_retrieved_has_components_default():
    from doclens.types import Retrieved, Chunk
    r = Retrieved(chunk=Chunk("d-0","d",1,0,"x"), score=0.5)
    assert r.components == {}
