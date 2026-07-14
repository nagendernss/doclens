import pytest

from evals.metrics import faithful, load_gold, mrr, recall_at_k


def test_recall():
    assert recall_at_k(["a", "b"], ["b"]) == 1.0
    assert recall_at_k(["a"], ["z"]) == 0.0


def test_mrr():
    assert mrr(["x", "rel", "y"], ["rel"]) == 0.5
    assert mrr(["x"], ["rel"]) == 0.0


def test_faithful():
    assert faithful("Refund in 30 days [p.2].", [2], [2, 5], ["30 days"]) is True
    assert faithful("Refund [p.9].", [9], [2], ["refund"]) is False   # cites unretrieved page
    assert faithful("No facts here [p.2].", [2], [2], ["missing fact"]) is False


def test_load_gold_validates(tmp_path):
    bad = tmp_path / "g.yaml"
    bad.write_text("cases:\n  - id: x\n    doc: a.md\n")
    with pytest.raises(ValueError, match="x"):
        load_gold(str(bad))


def test_seed_gold_loads():
    cases = load_gold("evals/gold.yaml")
    assert len(cases) >= 30
    assert sum(1 for c in cases if not c["answerable"]) >= 5
