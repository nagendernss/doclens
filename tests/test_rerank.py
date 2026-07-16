from doclens.rerank import llm_rerank, RERANK_INPUT_CHARS
from doclens.types import Retrieved, Chunk, Usage


def _cands(n):
    return [Retrieved(Chunk(f"d-{i:04d}", "d", i + 1, i, f"text {i}"), 0.5) for i in range(n)]


class FakeChat:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete(self, messages, model):
        self.calls += 1
        return self.reply, Usage(10, 5)


def test_valid_json_reorders_and_sets_rank():
    chat = FakeChat("[3, 1, 2]")
    out, usage = llm_rerank(chat, "m", "q", _cands(3), top_k=3)
    assert [r.chunk.seq for r in out] == [2, 0, 1]        # 1-based → 0-based
    assert out[0].components["rerank_rank"] == 1
    assert usage.input_tokens == 10


def test_garbage_falls_back_to_original_topk():
    chat = FakeChat("the passages are all great")
    out, _ = llm_rerank(chat, "m", "q", _cands(4), top_k=2)
    assert [r.chunk.seq for r in out] == [0, 1]           # original order
    assert chat.calls == 1                                 # called, then fell back


def test_missing_ids_appended_invalid_dropped():
    chat = FakeChat("[2, 99, 2, 1]")                       # dup 2, invalid 99, missing 3
    out, _ = llm_rerank(chat, "m", "q", _cands(3), top_k=3)
    assert sorted(r.chunk.seq for r in out) == [0, 1, 2]  # full coverage
    assert out[0].chunk.seq == 1                           # id 2 → idx 1 first


def test_single_candidate_makes_no_call():
    chat = FakeChat("[1]")
    out, usage = llm_rerank(chat, "m", "q", _cands(1), top_k=5)
    assert chat.calls == 0 and len(out) == 1 and usage.input_tokens == 0


# --- Additional hardening tests (adversarial self-review of the failure path) ---

def test_empty_candidates_makes_no_call():
    chat = FakeChat("[1]")
    out, usage = llm_rerank(chat, "m", "q", [], top_k=5)
    assert chat.calls == 0
    assert out == []
    assert usage.input_tokens == 0


def test_fallback_leaves_rerank_rank_unset():
    chat = FakeChat("nope, not json at all")
    out, _ = llm_rerank(chat, "m", "q", _cands(3), top_k=3)
    assert chat.calls == 1
    assert all("rerank_rank" not in r.components for r in out)


def test_success_does_not_mutate_or_alias_original_candidates():
    cands = _cands(3)
    chat = FakeChat("[3, 1, 2]")
    out, _ = llm_rerank(chat, "m", "q", cands, top_k=3)
    # originals untouched: no shared-state corruption from setting rerank_rank
    assert all(c.components == {} for c in cands)
    # returned items are distinct objects, not the same Retrieved instances
    assert all(r is not c for r, c in zip(out, cands))


class RecordingChat:
    """Fake chat that records the prompt it was sent, for shape assertions."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.last_prompt = None

    def complete(self, messages, model):
        self.calls += 1
        self.last_prompt = messages[0]["content"]
        return self.reply, Usage(10, 5)


def test_prompt_truncates_candidate_text_to_400_chars():
    long_text = "x" * (RERANK_INPUT_CHARS + 100)
    cands = [
        Retrieved(Chunk("d-0000", "d", 1, 0, long_text), 0.5),
        Retrieved(Chunk("d-0001", "d", 2, 1, "short"), 0.5),
    ]
    chat = RecordingChat("[1, 2]")
    llm_rerank(chat, "m", "q", cands, top_k=2)
    assert ("x" * RERANK_INPUT_CHARS) in chat.last_prompt
    assert ("x" * (RERANK_INPUT_CHARS + 1)) not in chat.last_prompt
