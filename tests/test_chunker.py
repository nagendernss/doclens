from doclens.chunker import chunk_document
from doclens.types import Document, PageText


def make_doc(texts):
    return Document(doc_id="d1", title="t", source="s",
                    pages=[PageText(i + 1, t) for i, t in enumerate(texts)])


def test_short_page_single_chunk():
    doc = make_doc(["Heading line\nBody text here."])
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1-0000"
    assert chunks[0].page == 1 and chunks[0].heading == "Heading line"


def test_long_page_overlapping_windows():
    sentence = "Alpha bravo charlie delta echo foxtrot golf hotel. "
    doc = make_doc([sentence * 100])  # ~5200 chars
    chunks = chunk_document(doc, target_chars=2000, overlap=0.15)
    assert len(chunks) >= 3
    # windows overlap: next chunk starts before previous ends
    joined = doc.pages[0].text
    first_end = joined.find(chunks[1].text[:40])
    assert 0 < first_end < 2000
    # sentence snap: every non-final chunk ends at a sentence boundary
    for c in chunks[:-1]:
        assert c.text.rstrip().endswith((".", "?", "!"))


def test_seq_spans_pages():
    doc = make_doc(["one. " * 500, "two. " * 500])
    chunks = chunk_document(doc)
    assert [c.page for c in chunks[:1]][0] == 1
    assert chunks[-1].page == 2
    seqs = [c.seq for c in chunks]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
