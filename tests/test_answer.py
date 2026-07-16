from doclens.answer import answer_question, RETRIEVAL_MODES
from doclens.hybrid import HybridIndex
from doclens.trace import Tracer
from doclens.types import Chunk, Usage


class FakeChat:
    def __init__(self, reply="Answer [p.1]"):
        self.reply = reply
        self.calls = []

    def complete(self, messages, model):
        self.calls.append(messages)
        return self.reply, Usage(10, 5)


class FakeEmbedder:
    def __init__(self, vec):
        self.vec = vec

    def embed(self, texts, model):
        return [self.vec for _ in texts]


class FakeEmb:
    """Embedder stub that always returns the same fixed vector (ignores input text)."""

    def embed(self, texts, model):
        return [[1.0, 0.0] for _ in texts]


class OrthoEmb:
    """Embedder stub whose vector is perpendicular to every doc vector in _idx() -> cos ~0."""

    def embed(self, texts, model):
        return [[0.0, 1.0] for _ in texts]


def make_index():
    idx = HybridIndex()
    idx.add(
        [Chunk("c0", "d", 2, 0, "The refund window is 30 days."),
         Chunk("c1", "d", 5, 1, "Contact support by email.")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    return idx


def _idx():
    idx = HybridIndex()
    idx.add(
        [Chunk("d-0000", "d", 1, 0, "alpha content"), Chunk("d-0001", "d", 2, 1, "beta content")],
        [[1.0, 0.0], [0.9, 0.1]],
    )
    return idx


# The four legacy tests below assert on the exact shape of chat.calls[0], so they pin
# retrieval_mode="hybrid" to keep the (only) chat call the generate call.
# test_low_score_refuses_without_llm pins retrieval_mode="hybrid_rerank" explicitly to
# prove refusal short-circuits before rerank too (independent of the default mode).

def test_grounded_answer_with_citations():
    chat = FakeChat("Refunds are allowed within 30 days [p.2].")
    res = answer_question(chat, "m", FakeEmbedder([1.0, 0.0]), "e", make_index(),
                          "what is the refund window?", retrieval_mode="hybrid")
    assert res.refused is False
    assert res.citations == [2]
    assert res.retrieved[0].chunk.chunk_id == "c0"
    context = chat.calls[0][-1]["content"]
    assert "[p.2]" in context
    assert "The refund window is 30 days." in context


def test_low_score_refuses_without_llm():
    chat = FakeChat("should never be called")
    # hybrid_rerank is the most-stage mode; refusal must still short-circuit
    # before ANY chat call (both rerank and generate) when the top dense cosine
    # is below threshold.
    res = answer_question(chat, "m", FakeEmbedder([0.0, 0.0]), "e", make_index(),
                          "unrelated question", retrieval_mode="hybrid_rerank")
    assert res.refused is True and chat.calls == []
    assert res.answer.startswith("Not in the document")


def test_model_refusal_detected():
    chat = FakeChat("Not in the document. The context never mentions pricing.")
    res = answer_question(chat, "m", FakeEmbedder([1.0, 0.0]), "e", make_index(), "pricing?",
                          retrieval_mode="hybrid")
    assert res.refused is True and res.citations == []


def test_history_injected_before_context_question():
    chat = FakeChat("Their degree is B.Tech [p.1].")
    history = [
        {"question": "what is the name?", "answer": "Nagender [p.1]."},
        {"question": "their email?", "answer": "n@x.com [p.1]."},
    ]
    res = answer_question(chat, "m", FakeEmbedder([1.0, 0.0]), "e", make_index(),
                          "what did they study?", history=history, retrieval_mode="hybrid")
    msgs = chat.calls[0]
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "what is the name?"}
    assert msgs[2] == {"role": "assistant", "content": "Nagender [p.1]."}
    assert msgs[3] == {"role": "user", "content": "their email?"}
    assert msgs[4] == {"role": "assistant", "content": "n@x.com [p.1]."}
    assert msgs[5]["role"] == "user"
    assert "what did they study?" in msgs[5]["content"]  # final = context + question
    assert res.answer.startswith("Their degree")


def test_no_history_unchanged():
    chat = FakeChat("A [p.1].")
    answer_question(chat, "m", FakeEmbedder([1.0, 0.0]), "e", make_index(), "q", history=None,
                    retrieval_mode="hybrid")
    msgs = chat.calls[0]
    assert len(msgs) == 2  # system + context/question only


def test_dense_mode_no_rerank_call_still_answers():
    chat = FakeChat()
    res = answer_question(chat, "m", FakeEmb(), "e", _idx(), "alpha", k=2,
                          retrieval_mode="dense")
    assert not res.refused and len(chat.calls) == 1     # generate only


def test_hybrid_rerank_invokes_reranker():
    class TwoReply:
        def __init__(self):
            self.n = 0

        def complete(self, messages, model):
            self.n += 1
            return ("[2,1]" if self.n == 1 else "Answer [p.1]"), Usage(5, 2)

    c = TwoReply()
    res = answer_question(c, "m", FakeEmb(), "e", _idx(), "alpha", k=2,
                          retrieval_mode="hybrid_rerank")
    assert c.n == 2 and not res.refused
    assert res.usage.input_tokens == 10                 # 5 rerank + 5 generate


def test_low_cosine_refuses_in_every_mode():
    for mode in RETRIEVAL_MODES:
        res = answer_question(FakeChat(), "m", OrthoEmb(), "e", _idx(), "zzz",
                              retrieval_mode=mode)
        assert res.refused, mode


def test_tracer_records_stage_spans():
    t = Tracer()
    answer_question(FakeChat(), "m", FakeEmb(), "e", _idx(), "alpha",
                    retrieval_mode="hybrid", tracer=t)
    names = [s.name for s in t.trace.spans]
    assert "embed" in names and "retrieve" in names and "generate" in names
    assert "rerank" not in names                        # hybrid (no rerank)


# --- Additional hardening tests (adversarial self-review of the flow contract) ---

def test_refusal_retrieved_truncated_to_k_not_empty():
    # The UI still needs to show what was searched on a refusal, so `retrieved`
    # must be the (possibly larger) candidate pool truncated to k, never empty.
    res = answer_question(FakeChat(), "m", OrthoEmb(), "e", _idx(), "zzz", k=1)
    assert res.refused
    assert 0 < len(res.retrieved) <= 1


def test_refusal_records_only_embed_and_retrieve_spans():
    # Even in hybrid_rerank (the mode with the most stages), refusal must
    # short-circuit before both the rerank and generate spans are opened.
    t = Tracer()
    answer_question(FakeChat(), "m", OrthoEmb(), "e", _idx(), "zzz",
                    retrieval_mode="hybrid_rerank", tracer=t)
    names = [s.name for s in t.trace.spans]
    assert names == ["embed", "retrieve"]


def test_hybrid_rerank_mode_records_rerank_span():
    # Complements test_tracer_records_stage_spans (hybrid, no rerank): the
    # default mode must record all four spans, in stage order.
    t = Tracer()
    answer_question(FakeChat(), "m", FakeEmb(), "e", _idx(), "alpha",
                    retrieval_mode="hybrid_rerank", tracer=t)
    names = [s.name for s in t.trace.spans]
    assert names == ["embed", "retrieve", "rerank", "generate"]
