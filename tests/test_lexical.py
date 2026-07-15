from doclens.lexical import BM25Index, _tokenize
from doclens.types import Chunk


def _c(i, text):
    return Chunk(chunk_id=f"d-{i:04d}", doc_id="d", page=1, seq=i, text=text)


def test_tokenize_drops_stopwords_and_short():
    toks = _tokenize("The quick BROWN fox, a fox!")
    assert "the" not in toks and "a" not in toks
    assert toks.count("fox") == 2 and "brown" in toks


def test_exact_term_outranks_partial_overlap():
    # Both chunks share "energy" so both enter scoring and genuinely compete;
    # chunk 0 also matches "mitochondria" so it must outrank the partial hit.
    idx = BM25Index()
    idx.add([_c(0, "mitochondria powerhouse energy conversion"),
             _c(1, "cellular energy production process")])
    ranked = idx.rank("mitochondria energy")
    assert len(ranked) == 2                     # both competed, not a 1-element list
    assert ranked[0][0] == 0                     # full match beats partial match
    assert all(s > 0 for _, s in ranked)


def test_rank_tiebreak_is_index_ascending_not_insertion_order():
    # Disjoint single-term chunks → equal scores. Query orders "beta" first so
    # chunk 1 enters the score dict before chunk 0; a stability-only sort key
    # (-score with no idx tiebreak) would yield [1, 0]. Correct (-score, idx) → [0, 1].
    idx = BM25Index()
    idx.add([_c(0, "alpha"), _c(1, "beta")])
    ranked = idx.rank("beta alpha")
    assert [i for i, _ in ranked] == [0, 1]               # idx asc tiebreak, not insertion order


def test_empty_index_and_empty_query():
    assert BM25Index().rank("anything") == []
    idx = BM25Index()
    idx.add([_c(0, "hello world")])
    assert idx.rank("") == []
    assert idx.rank("nonexistentterm") == []


def test_len_and_incremental_add():
    idx = BM25Index()
    idx.add([_c(0, "a cat")])
    idx.add([_c(1, "a dog")])
    assert len(idx) == 2
    assert idx.rank("dog")[0][0] == 1
