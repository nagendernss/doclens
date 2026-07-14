import json

from doclens.chunker import chunk_document
from doclens.ingest import ingest_file
from doclens.types import Usage, fingerprint
from evals.report import splice_readme, summarize, to_markdown
from evals.run import run_eval


class FakeChat:
    def complete(self, messages, model):
        return "The spec version is 3 [p.1].", Usage(5, 2)


class FakeEmb:
    def embed(self, texts, model):
        return [[1.0, 0.0] for _ in texts]


def gold_cases():
    """Gold cases with computed fingerprints from real corpus."""
    # Compute first-chunk fingerprint of rfc-style-spec.md
    doc = ingest_file("evals/corpus/rfc-style-spec.md")
    chunks = chunk_document(doc)
    first_chunk_fp = fingerprint(chunks[0].doc_id, chunks[0].page, chunks[0].text)

    return [
        {"id": "g1", "doc": "rfc-style-spec.md", "question": "what version?",
         "relevant_fps": [first_chunk_fp], "expected_facts": ["version is 3"],
         "answerable": True},
        {"id": "g2", "doc": "rfc-style-spec.md", "question": "who is the CEO?",
         "relevant_fps": [], "expected_facts": [], "answerable": False},
    ]


def test_run_eval_and_summarize(tmp_path, monkeypatch):
    out = tmp_path / "r.json"
    cases = gold_cases()
    results = run_eval(["gemini-3.1-flash-lite"], cases, str(out),
                       embedder_factory=lambda **kw: (FakeEmb(), "e"),
                       chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    records = results["records"]
    assert len(records) == 2
    g1 = next(r for r in records if r["case_id"] == "g1")
    assert g1["error"] is None and isinstance(g1["recall5"], float)
    g2 = next(r for r in records if r["case_id"] == "g2")
    assert g2["recall5"] is None and g2["refused_correctly"] in (True, False)
    # resume: second run does nothing new
    n_before = len(json.loads(out.read_text())["records"])
    run_eval(["gemini-3.1-flash-lite"], cases, str(out),
             embedder_factory=lambda **kw: (FakeEmb(), "e"),
             chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    assert len(json.loads(out.read_text())["records"]) == n_before


def test_report_table_and_splice():
    records = [
        {"model": "m", "case_id": "a", "recall5": 1.0, "mrr": 1.0, "faithful": True,
         "refused_correctly": None, "latency_s": 1.0, "input_tokens": 1,
         "output_tokens": 1, "error": None},
        {"model": "m", "case_id": "b", "recall5": None, "mrr": None, "faithful": None,
         "refused_correctly": True, "latency_s": 2.0, "input_tokens": 1,
         "output_tokens": 1, "error": None},
    ]
    (s,) = summarize(records)
    assert s["recall_at_5"] == 1.0 and s["refusal_acc"] == 1.0
    md = to_markdown([s])
    assert "| Model | Recall@5 |" in md and "| m |" in md
    spliced = splice_readme("a\n<!-- evals:start -->\nold\n<!-- evals:end -->\nb", "T")
    assert "T" in spliced and "old" not in spliced
