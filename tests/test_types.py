from doclens.types import Chunk, Document, PageText, Retrieved, Usage, fingerprint


def test_usage_add():
    assert (Usage(1, 2) + Usage(3, 4)) == Usage(4, 6)


def test_fingerprint_normalizes():
    fp = fingerprint("d1", 3, "  The QUICK, brown fox—jumps over the lazy dog today!  ")
    assert fp == "d1|p3|the quick brown foxjumps over the lazy dog"


def test_shapes():
    d = Document(doc_id="d1", title="T", source="file.pdf", pages=[PageText(1, "hi")])
    c = Chunk(chunk_id="c1", doc_id="d1", page=1, seq=0, text="hi")
    r = Retrieved(chunk=c, score=0.5)
    assert d.pages[0].page == 1 and r.chunk.chunk_id == "c1" and c.heading == ""
