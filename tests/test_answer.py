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


def test_history_injected_before_context_question():
    chat = FakeChat("Their degree is B.Tech [p.1].")
    history = [
        {"question": "what is the name?", "answer": "Nagender [p.1]."},
        {"question": "their email?", "answer": "n@x.com [p.1]."},
    ]
    res = answer_question(chat, "m", FakeEmbedder([1.0, 0.0]), "e", make_index(),
                          "what did they study?", history=history)
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
    answer_question(chat, "m", FakeEmbedder([1.0, 0.0]), "e", make_index(), "q", history=None)
    msgs = chat.calls[0]
    assert len(msgs) == 2  # system + context/question only
