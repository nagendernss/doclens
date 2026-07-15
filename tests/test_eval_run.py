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


class CountingEmb:
    """Embedder that counts how many times embed() is called."""
    def __init__(self):
        self.call_count = 0

    def embed(self, texts, model):
        self.call_count += 1
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
                       modes=("hybrid_rerank",),
                       embedder_factory=lambda **kw: (FakeEmb(), "e"),
                       chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    records = results["records"]
    assert len(records) == 2
    g1 = next(r for r in records if r["case_id"] == "g1")
    assert g1["error"] is None and isinstance(g1["recall5"], float)
    assert g1["mode"] == "hybrid_rerank"
    g2 = next(r for r in records if r["case_id"] == "g2")
    assert g2["recall5"] is None and g2["refused_correctly"] in (True, False)
    # resume: second run does nothing new
    n_before = len(json.loads(out.read_text())["records"])
    run_eval(["gemini-3.1-flash-lite"], cases, str(out),
             modes=("hybrid_rerank",),
             embedder_factory=lambda **kw: (FakeEmb(), "e"),
             chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    assert len(json.loads(out.read_text())["records"]) == n_before


def test_run_eval_modes_axis(tmp_path):
    """1 model x 3 modes x 1 case -> one record per mode, each tagged and error-free."""
    out = tmp_path / "r.json"
    case = gold_cases()[:1]  # answerable case "g1" only
    results = run_eval(["gemini-3.1-flash-lite"], case, str(out),
                       modes=("dense", "hybrid", "hybrid_rerank"),
                       embedder_factory=lambda **kw: (FakeEmb(), "e"),
                       chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    records = results["records"]
    assert len(records) == 3
    assert {r["mode"] for r in records} == {"dense", "hybrid", "hybrid_rerank"}
    assert all(r["case_id"] == "g1" for r in records)
    # HybridIndex flows cleanly through answer_question in every mode (no
    # VectorIndex-attribute error, no unhandled rerank-parse exception).
    assert all(r["error"] is None for r in records)


def test_run_eval_resume_keys_on_mode_triple(tmp_path):
    """Resume dedupes on (model, mode, case_id): a pre-seeded mode isn't re-run."""
    out = tmp_path / "r.json"
    case = gold_cases()[:1]
    seed = {
        "records": [{
            "model": "gemini-3.1-flash-lite", "mode": "dense", "case_id": "g1",
            "recall5": 1.0, "mrr": 1.0, "faithful": True, "refused_correctly": None,
            "latency_s": 0.1, "input_tokens": 1, "output_tokens": 1, "error": None,
        }],
        "metadata": {},
    }
    out.write_text(json.dumps(seed))

    results = run_eval(["gemini-3.1-flash-lite"], case, str(out),
                       modes=("dense", "hybrid"),
                       embedder_factory=lambda **kw: (FakeEmb(), "e"),
                       chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    records = results["records"]
    # The pre-seeded (model, dense, g1) triple must not be duplicated; only
    # (model, hybrid, g1) is new.
    assert len(records) == 2
    dense_records = [r for r in records if r["mode"] == "dense"]
    hybrid_records = [r for r in records if r["mode"] == "hybrid"]
    assert len(dense_records) == 1
    assert len(hybrid_records) == 1


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


def test_corrupt_nondict_starts_fresh(tmp_path):
    """Test that corrupt JSON (wrong shape) recovers gracefully without crash."""
    out = tmp_path / "r.json"
    cases = gold_cases()

    # Test case 1: JSON array at top level
    out.write_text("[]")
    results = run_eval(["gemini-3.1-flash-lite"], cases, str(out),
                       modes=("hybrid_rerank",),
                       embedder_factory=lambda **kw: (FakeEmb(), "e"),
                       chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    assert isinstance(results, dict)
    assert "records" in results
    parsed = json.loads(out.read_text())
    assert isinstance(parsed, dict) and "records" in parsed
    assert len(parsed["records"]) == 2

    # Test case 2: JSON null at top level
    out.write_text("null")
    results = run_eval(["gemini-3.1-flash-lite"], cases, str(out),
                       modes=("hybrid_rerank",),
                       embedder_factory=lambda **kw: (FakeEmb(), "e"),
                       chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    assert isinstance(results, dict)
    assert "records" in results
    parsed = json.loads(out.read_text())
    assert isinstance(parsed, dict) and "records" in parsed
    assert len(parsed["records"]) == 2


def test_resume_noop_skips_embedding(tmp_path):
    """Test that embedding is not called when all cases are already in resume_set."""
    out = tmp_path / "r.json"
    cases = gold_cases()

    # First run: build results
    embedder1 = CountingEmb()
    results = run_eval(["gemini-3.1-flash-lite"], cases, str(out),
                       modes=("hybrid_rerank",),
                       embedder_factory=lambda **kw: (embedder1, "e"),
                       chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    n_records_1 = len(results["records"])
    embed_calls_1 = embedder1.call_count
    assert n_records_1 == 2
    assert embed_calls_1 > 0  # Should have embedded at least once

    # Second run: all cases already in results, should not call embed()
    embedder2 = CountingEmb()
    results = run_eval(["gemini-3.1-flash-lite"], cases, str(out),
                       modes=("hybrid_rerank",),
                       embedder_factory=lambda **kw: (embedder2, "e"),
                       chat_factory=lambda m, **kw: (FakeChat(), m), sleep_s=0)
    n_records_2 = len(results["records"])
    embed_calls_2 = embedder2.call_count
    assert n_records_2 == n_records_1  # No new records added
    assert embed_calls_2 == 0  # No embedding calls on full resume
