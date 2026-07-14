import pytest
from pypdf import PdfWriter

from doclens.ingest import MAX_PDF_BYTES, IngestError, ingest_file, ingest_pdf_bytes, ingest_text


def make_pdf(pages_text):
    """Build a real PDF in memory with one text page per entry (pypdf only)."""
    import io

    from pypdf.generic import (ArrayObject, DictionaryObject, NameObject,
                               NumberObject, StreamObject)
    writer = PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
        })
        stream = StreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        stream[NameObject("/Length")] = NumberObject(len(stream.get_data()))
        page[NameObject("/Contents")] = ArrayObject([writer._add_object(stream)])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_pages_extracted():
    data = make_pdf(["hello page one", "second page here"])
    doc = ingest_pdf_bytes(data, "sample.pdf")
    assert len(doc.pages) == 2
    assert "hello page one" in doc.pages[0].text
    assert doc.pages[1].page == 2
    assert len(doc.doc_id) == 12


def test_pdf_size_cap():
    with pytest.raises(IngestError, match="10"):
        ingest_pdf_bytes(b"x" * (MAX_PDF_BYTES + 1), "big.pdf")


def test_pdf_garbage_raises():
    with pytest.raises(IngestError):
        ingest_pdf_bytes(b"not a pdf at all", "junk.pdf")


def test_ingest_text_paginates():
    doc = ingest_text("A" * 7000, "notes.txt")
    assert [p.page for p in doc.pages] == [1, 2, 3]
    assert doc.title == "notes.txt"


def test_ingest_file_txt(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world")
    doc = ingest_file(str(f))
    assert doc.pages[0].text == "hello world"


def test_pdf_page_cap():
    """Regression: PDFs exceeding 300-page cap must raise IngestError."""
    import io
    writer = PdfWriter()
    for _ in range(301):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    data = buf.getvalue()
    with pytest.raises(IngestError, match="301 pages"):
        ingest_pdf_bytes(data, "too-many-pages.pdf")


def test_pdf_no_text_raises_ocr_hint():
    """Regression: PDFs with no extractable text (blank pages) hint at OCR."""
    import io
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    data = buf.getvalue()
    with pytest.raises(IngestError, match="scanned"):
        ingest_pdf_bytes(data, "blank-pages.pdf")
