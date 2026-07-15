from doclens.lexical import BM25Index, _tokenize
from doclens.types import Chunk


def _c(i, text):
    return Chunk(chunk_id=f"d-{i:04d}", doc_id="d", page=1, seq=i, text=text)


def test_tokenize_drops_stopwords_and_short():
    toks = _tokenize("The quick BROWN fox, a fox!")
    assert "the" not in toks and "a" not in toks
    assert toks.count("fox") == 2 and "brown" in toks


def test_exact_term_outranks_paraphrase():
    idx = BM25Index()
    idx.add([_c(0, "The mitochondria is the powerhouse of the cell."),
             _c(1, "Cellular energy production occurs in specialized organelles.")])
    ranked = idx.rank("mitochondria")
    assert ranked[0][0] == 0            # exact lexical hit first
    assert all(s > 0 for _, s in ranked)


def test_rank_sorted_and_tiebroken_by_index():
    idx = BM25Index()
    idx.add([_c(0, "alpha beta"), _c(1, "alpha beta")])   # identical → same score
    ranked = idx.rank("alpha")
    assert [i for i, _ in ranked] == [0, 1]               # idx asc tiebreak


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
