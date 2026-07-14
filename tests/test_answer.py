from doclens.answer import answer_question
from doclens.index import VectorIndex
from doclens.types import Chunk, Usage


class FakeChat:
    def __init__(self, reply):
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


def make_index():
    idx = VectorIndex()
    idx.add(
        [Chunk("c0", "d", 2, 0, "The refund window is 30 days."),
         Chunk("c1", "d", 5, 1, "Contact support by email.")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    return idx


def test_grounded_answer_with_citations():
    chat = FakeChat("Refunds are allowed within 30 days [p.2].")
    res = answer_question(chat, "m", FakeEmbedder([1.0, 0.0]), "e", make_index(),
                          "what is the refund window?")
    assert res.refused is False
    assert res.citations == [2]
    assert res.retrieved[0].chunk.chunk_id == "c0"
    context = chat.calls[0][-1]["content"]
    assert "[p.2]" in context
    assert "The refund window is 30 days." in context


def test_low_score_refuses_without_llm():
    chat = FakeChat("should never be called")
    res = answer_question(chat, "m", FakeEmbedder([0.0, 0.0]), "e", make_index(),
                          "unrelated question")
    assert res.refused is True and chat.calls == []
    assert res.answer.startswith("Not in the document")


def test_model_refusal_detected():
    chat = FakeChat("Not in the document. The context never mentions pricing.")
    res = answer_question(chat, "m", FakeEmbedder([1.0, 0.0]), "e", make_index(), "pricing?")
    assert res.refused is True and res.citations == []
