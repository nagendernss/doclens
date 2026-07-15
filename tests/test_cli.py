from unittest.mock import MagicMock, patch

from doclens.cli import main
from doclens.types import AnswerResult, Chunk, Document, PageText, Retrieved, Usage


def fake_doc():
    return Document("d1", "Title", "s.pdf", [PageText(1, "text")])


def fake_answer():
    ch = Chunk("c0", "d1", 2, 0, "chunk text")
    return AnswerResult("Answer [p.2].", [2], [Retrieved(ch, 0.9)], False, "m", Usage(9, 3))


@patch("doclens.cli.answer_question", return_value=fake_answer())
@patch("doclens.cli.HybridIndex")
@patch("doclens.cli.get_embedder", return_value=(MagicMock(embed=lambda t, m: [[1.0]] * len(t)), "e"))
@patch("doclens.cli.get_chat", return_value=(MagicMock(), "m"))
@patch("doclens.cli.chunk_document", return_value=[Chunk("c0", "d1", 1, 0, "x")])
@patch("doclens.cli.ingest_file", return_value=fake_doc())
def test_ask_happy(mi, mc, mg, me, mh, ma, capsys):
    code = main(["ask", "doc.pdf", "what?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Title" in out and "Answer [p.2]." in out and "tokens=9+3" in out
    assert ma.call_args.kwargs["retrieval_mode"] == "hybrid_rerank"


@patch("doclens.cli.answer_question", return_value=fake_answer())
@patch("doclens.cli.HybridIndex")
@patch("doclens.cli.get_embedder", return_value=(MagicMock(embed=lambda t, m: [[1.0]] * len(t)), "e"))
@patch("doclens.cli.get_chat", return_value=(MagicMock(), "m"))
@patch("doclens.cli.chunk_document", return_value=[Chunk("c0", "d1", 1, 0, "x")])
@patch("doclens.cli.ingest_file", return_value=fake_doc())
def test_ask_mode_flag_threads_through_as_retrieval_mode(mi, mc, mg, me, mh, ma, capsys):
    code = main(["ask", "doc.pdf", "what?", "--mode", "dense"])
    assert code == 0
    assert ma.call_args.kwargs["retrieval_mode"] == "dense"


@patch("doclens.cli.get_embedder", return_value=(MagicMock(embed=lambda t, m: [[1.0]] * len(t)), "e"))
@patch("doclens.cli.get_chat", return_value=(MagicMock(), "m"))
@patch("doclens.cli.ingest_file", side_effect=__import__(
    "doclens.ingest", fromlist=["IngestError"]).IngestError("no extractable text"))
def test_ask_ingest_error_exit_1(mi, mc, me, capsys):
    code = main(["ask", "bad.pdf", "q"])
    assert code == 1 and "no extractable text" in capsys.readouterr().err


def test_models_lists(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "gemini-3.1-flash-lite *" in out
