# doclens Follow-up Chat + Clickable Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn each document's ask panel into a multi-turn conversation — follow-up questions understand prior turns (like repolens chat) — and make `[p.N]` citation pills clickable to scroll/highlight the retrieved chunk from that page.

**Architecture:** `answer_question` gains an optional `history` param (prior turns injected as user/assistant messages before the retrieval-context+question message; RAG still retrieves fresh on the current question). `/api/ask` sanitizes client-supplied history (last 6 turns, 1500-char answers, malformed dropped) and passes it through. The frontend keeps a per-doc conversation log in localStorage, sends accumulated turns, and renders citation pills that scroll to the matching retrieval-preview chunk.

**Tech Stack:** existing doclens core + web (Python/FastAPI + vanilla JS). No new deps.

## Global Constraints

- Branch `feat/chat` off main. Conventional commits. Every task committed.
- History sanitize (server): keep last **6** well-formed turns; truncate each `answer` to **1500** chars; a well-formed turn is a dict with non-empty-str `question` and non-empty-str `answer`; malformed dropped silently; missing/empty/non-list `history` ⇒ behave exactly as today.
- `answer_question(chat, chat_model, embedder, embed_model, index, question, k=5, history=None)` — history turns injected AFTER the system message, BEFORE the retrieval-context+question user message, as alternating `{"role":"user","content":q}` / `{"role":"assistant","content":a}`. History does NOT change retrieval (still `index.search(embed(question))`).
- SSE `/api/ask` contract unchanged (`retrieval` then `answer`); each follow-up costs 1 question rate-unit.
- No live network in tests (fakes/TestClient; current suite = 135, all stay green).
- CLI unaffected (doesn't pass history).
- Frontend security unchanged: all model output via esc()/escAttr(); no new innerHTML sinks; citation pills already digits-only.
- localStorage per-doc convo: extend `doclens.docs.v1` or add `doclens.convos.v1` = `{doc_id: [{q, answer, citations, retrieval, ts}]}`, cap 40 turns/doc, oldest evicted, corrupt JSON → fresh.

---

### Task 1: Agent history injection

**Files:** Modify `doclens/answer.py` (function `answer_question`); Test `tests/test_answer.py` (append).

**Interfaces:**
- Consumes: existing `answer_question`.
- Produces: `answer_question(..., history: list[dict] | None = None)`. Each history item `{"question": str, "answer": str}` → two messages injected between system and the final context+question user message. Order preserved. `None`/`[]` ⇒ byte-identical to today. Retrieval + refusal logic unchanged. Server passes pre-sanitized history (agent does NOT re-validate).

- [ ] **Step 1: Write the failing test** (append to `tests/test_answer.py`):
```python
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
```
(Reuse the existing `FakeChat`/`FakeEmbedder`/`make_index` helpers in this file. Note `FakeChat.complete(messages, model)` records `messages` in `self.calls` — confirm the helper does this; if it currently records differently, adapt the assertions to the actual recorded shape, keeping the ordering contract.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_answer.py -v`
Expected: FAIL (`answer_question() got an unexpected keyword argument 'history'`)

- [ ] **Step 3: Write minimal implementation**

In `answer_question`, after building `qvec`/`retrieved` and BEFORE constructing `messages`, keep the refusal short-circuit exactly as-is (refusal must NOT depend on history). Then build messages with history injected:
```python
def answer_question(chat, chat_model, embedder, embed_model, index, question, k=5,
                    history=None):
    qvec = embedder.embed([question], embed_model)[0]
    retrieved = index.search(qvec, k=k)
    if not retrieved or retrieved[0].score < REFUSAL_THRESHOLD:
        return AnswerResult(answer=REFUSAL_TEXT, citations=[], retrieved=retrieved,
                            refused=True, model=chat_model, usage=Usage())
    context = "\n\n".join(f"[p.{r.chunk.page}] {r.chunk.text}" for r in retrieved)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user",
                     "content": f"Context chunks:\n\n{context}\n\nQuestion: {question}"})
    text, usage = chat.complete(messages, chat_model)
    ...
```
(Keep the citation parse + refused detection tail exactly as today.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_answer.py -v` — Expected: all pass. Full suite: 137 passed. `ruff check .` clean.

- [ ] **Step 5: Commit**

```bash
git add doclens/answer.py tests/test_answer.py
git commit -m "feat: answer_question accepts prior-turn history for follow-ups"
```

---

### Task 2: Server history sanitize + pass-through

**Files:** Modify `doclens/server.py` (ask endpoint); Test `tests/test_server_ask.py` (append).

**Interfaces:**
- Produces: module-level `sanitize_history(raw) -> list[dict]` — `[]` for non-list; keep dicts with non-empty-str `question` AND non-empty-str `answer`; truncate `answer` to 1500 chars; return LAST 6, original order. Ask endpoint reads `body.get("history")`, sanitizes once, passes `history=` into `answer_question` (inside the worker `work()`).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_server_ask.py`):
```python
from doclens.server import sanitize_history


def test_sanitize_history_rules():
    raw = ([{"question": f"q{i}", "answer": "a" * 2000} for i in range(8)]
           + ["junk", {"question": "", "answer": "x"}, {"question": "ok"},
              {"question": "x", "answer": "   "}, {"question": "last", "answer": "short"}])
    out = sanitize_history(raw)
    assert len(out) == 6
    assert out[-1] == {"question": "last", "answer": "short"}
    assert all(len(t["answer"]) <= 1500 for t in out)
    assert out[0]["question"] == "q3"


def test_sanitize_history_non_list():
    assert sanitize_history(None) == []
    assert sanitize_history("x") == []
    assert sanitize_history({"question": "q", "answer": "a"}) == []


def test_ask_passes_sanitized_history(client_with_doc, monkeypatch):
    # client_with_doc: the existing fixture that seeds a session doc + monkeypatches
    # answer_question. Replace the monkeypatch to capture history.
    import doclens.server as srv
    seen = {}

    def capture(chat, chat_model, embedder, embed_model, index, question, k=5, history=None):
        seen["history"] = history
        return _fake_answer_result()  # reuse the file's existing fake AnswerResult builder

    monkeypatch.setattr(srv, "answer_question", capture)
    r = client_with_doc.post("/api/ask", json={
        "doc_id": SEEDED_DOC_ID, "question": "follow-up?",
        "history": [{"question": "q1", "answer": "a1"}, "junk"],
    })
    events = sse_events(r.text)
    assert events[-1][0] == "answer"
    assert seen["history"] == [{"question": "q1", "answer": "a1"}]
```
(Adapt `client_with_doc`, `SEEDED_DOC_ID`, `_fake_answer_result`, `sse_events` to the actual fixtures/helpers already in `tests/test_server_ask.py` — read the file first and mirror its existing seeding pattern.)

- [ ] **Step 2: Run → FAIL** (`ImportError: sanitize_history`).

- [ ] **Step 3: Implement** — add to `doclens/server.py`:
```python
MAX_HISTORY_TURNS = 6
MAX_HISTORY_ANSWER_CHARS = 1500


def sanitize_history(raw) -> list[dict]:
    if not isinstance(raw, list):
        return []
    turns = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q, a = item.get("question"), item.get("answer")
        if not isinstance(q, str) or not q.strip() or not isinstance(a, str) or not a.strip():
            continue
        turns.append({"question": q, "answer": a[:MAX_HISTORY_ANSWER_CHARS]})
    return turns[-MAX_HISTORY_TURNS:]
```
In the ask handler, read `history = sanitize_history(body.get("history"))` alongside the other body fields, and pass `history=history` into the `answer_question(...)` call inside `work()`.

- [ ] **Step 4: Run → pass**; full suite 140 passed; ruff clean.

- [ ] **Step 5: Commit** `feat: /api/ask sanitizes and forwards follow-up history`.

---

### Task 3: Frontend conversation + clickable citations

**Files:** Modify `web/index.html`, `web/app.js`, `web/style.css`.

**Interfaces:** consumes `/api/ask` with `history`; existing SSE events. Reuse esc()/escAttr()/renderCitedAnswer().

Behavior contract:
- **Conversation log per doc:** the ask panel keeps a scrollable list of turns above the (pinned) question input. Each turn = user bubble (escaped text) + agent bubble (the existing cited-answer render + retrieval previews collapsed into a `<details>` "N sources"). Asking appends a turn; the input stays for the next follow-up.
- **History send:** on ask, send `history` = this doc's stored turns mapped to `[{question: t.q, answer: t.answer}]` (client sends all; server keeps last 6).
- **Persistence:** `doclens.convos.v1` = `{[doc_id]: [{q, answer, citations, retrieval, ts}]}`; cap 40 turns/doc (evict oldest); corrupt JSON → `{}`; load on doc select; clearing the doc list (existing "clear list") also clears convos.
- **Switching docs** shows that doc's own conversation.
- **Clickable citations:** in `renderCitedAnswer`, render each `[p.N]` as a `<button class="pill pill-cite" data-page="N">p.N</button>` (button, not span). Clicking scrolls the matching retrieval preview for THAT turn (the chunk(s) whose `page===N`) into view and briefly highlights it (add a `.flash` class, remove after ~1.2s). If no retrieval preview for that page in that turn, no-op (don't error). Keep pills digits-only → no injection. Non-`[p.N]` text still escaped.
- Refusal turns still render (no citations, "Not in the document"). "Document not found" error still shows the re-ingest banner and is NOT saved as a turn.
- Do NOT surface rate-limit remaining() as a number (carry-forward).

Steps:
- [ ] **Step 1:** Rework the ask-panel section of index.html + the ask/render logic in app.js + styles. Keep the doc switcher, ingest panel, and all Task-1..5 web behavior intact.
- [ ] **Step 2: Gates** — `node -c web/app.js`; `python -m pytest -q` (140, unchanged) + `ruff check .`; start server without key → `/` 200 contains "doclens", `/static/app.js` + `/static/style.css` 200. Paste outputs in report.
- [ ] **Step 3:** Headless-Chrome check (Chrome at "C:\Program Files\Google\Chrome\Application\chrome.exe"): seed `doclens.convos.v1` with a 2-turn convo + a mocked fetch; render; assert both turns present; click a `p.N` pill → assert the matching preview gets `.flash`; adversarial: inject `<img onerror>` into a stored answer/question/preview → assert inert. Paste evidence.
- [ ] **Step 4: Commit** `feat: per-doc follow-up conversation + clickable citation pills`.

---

### Task 4: Merge, deploy, live verify

- [ ] **Step 1:** Full gate on feat/chat (140 tests, ruff), merge `--no-ff` to main, push. Render auto-deploys.
- [ ] **Step 2:** Live verify on doclens-05fb.onrender.com: ingest a doc, ask a question, ask a FOLLOW-UP that references the prior answer ("and their email?" after "what's the name?") — confirm it uses context; click a citation pill → preview highlights; refresh → conversation persists (localStorage). curl a 2-turn-history /api/ask to confirm server passes history.
- [ ] **Step 3:** README: add "Chat mode" line (per-doc follow-ups, history in browser, clickable citations). Commit + push.
- [ ] **Step 4:** Ledger + memory update.

## Verification (whole plan)

1. 140 tests green, ruff clean.
2. Local + live: follow-up demonstrably uses prior context; citation pill click highlights source; convo persists across refresh; caps enforced.

## Out of scope

Cross-doc chat, retrieval over the conversation, token-streaming answers, editing past turns, server-side session persistence of chat.
