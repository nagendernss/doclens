# doclens Hybrid Retrieval + LLM Rerank + Tracing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two-stage retrieval (hybrid dense⊕BM25 RRF → LLM listwise rerank) plus a hand-built trace/waterfall layer, with a per-mode eval axis proving the gain.

**Architecture:** New `HybridIndex` wraps the dense `VectorIndex` + a hand-built `BM25Index`, fused by RRF; an LLM listwise reranker reorders the pool to top-k; every stage is a timed `Span` in a `Trace` surfaced as an SSE `trace` event + UI waterfall. Eval gains a `retrieval_mode` axis (`dense`|`hybrid`|`hybrid_rerank`).

**Tech Stack:** Python, numpy, httpx, FastAPI, PyYAML; raw REST Gemini adapters; vanilla JS. No new deps.

**Full design reference:** `docs/superpowers/specs/2026-07-15-doclens-retrieval-observability-upgrade.md`. Each task dispatch includes this spec path; the spec's section numbers are cited per task.

## Global Constraints

- **No new dependencies. No torch / sentence-transformers / Langfuse.** Must run on Render free tier (512 MB), no model downloads.
- **Raw httpx providers only** — reranker reuses `GeminiChat`.
- **Refusal `0.30` on max dense cosine, computed pre-rerank.** BM25/RRF scores never feed refusal.
- **SSE additive** — new `trace` event + new fields on `retrieval` chunks only.
- **Frontend security invariant** — `app.js` `innerHTML` only in `renderCitedAnswer`; waterfall/badges built with `textContent`/`createElement`/`style`. No new `innerHTML`.
- **`byo_key` never logged/echoed;** SSRF guard, rate limits, session-reset untouched.
- **Defaults:** pool `N=20`, `k=5`, RRF `k_const=60`, BM25 `k1=1.5`/`b=0.75`, rerank truncation `400` chars, `DEFAULT_RETRIEVAL_MODE="hybrid_rerank"`, trace ring cap `200`.
- **Carry-forward (from core ledger):** `fingerprint(doc_id,page,text)` uses the first 8 lowercased ASCII words → degenerates on non-ASCII text and on chunks sharing an opening. Any new corpus must be **English with distinct chunk openings**. Small corpus saturates recall@5 ~1.0 → MRR + faithfulness carry the retrieval-quality signal.
- **Style:** match existing module conventions (docstrings, `from __future__ import annotations`, ruff-clean). Run `python -m pytest -q` green before every commit.

---

## Task ordering & model tiers

1. BM25 (`lexical.py`) — cheap
2. RRF (`fusion.py`) — cheap
3. Tracer (`trace.py`) — cheap
4. `index.rank_all` + `Retrieved.components` — cheap
5. `HybridIndex` — standard
6. `llm_rerank` — standard
7. `answer.py` modes+rerank+tracer — standard
8. `sessions.py` HybridIndex + trace ring — cheap
9. `server.py` — standard
10. Frontend waterfall+badges — standard
11. `evals/run.py` modes — standard
12. `evals/report.py` Mode column — cheap
13. Gold cases + probe corpus + README — standard

---

### Task 1: BM25 lexical index

**Files:** Create `doclens/lexical.py`; Test `tests/test_lexical.py`. Spec §3.1.

**Interfaces:**
- Produces: `BM25Index` with `add(chunks: list[Chunk]) -> None`, `rank(query: str) -> list[tuple[int,float]]` (every chunk with score>0, sorted score desc then chunk_idx asc), `__len__`, ctor `(k1=1.5, b=0.75)`; module `_tokenize(text)->list[str]`, `STOPWORDS: frozenset[str]`. Chunk index space = add order (aligns with dense index).
- Consumes: `doclens.types.Chunk`.

- [ ] **Step 1: Failing tests**

```python
from doclens.lexical import BM25Index, _tokenize
from doclens.types import Chunk

def _c(i, text): return Chunk(chunk_id=f"d-{i:04d}", doc_id="d", page=1, seq=i, text=text)

def test_tokenize_drops_stopwords_and_short():
    toks = _tokenize("The quick BROWN fox, a fox!")
    assert "the" not in toks and "a" not in toks
    assert toks.count("fox") == 2 and "brown" in toks

def test_exact_term_outranks_paraphrase():
    idx = BM25Index()
    idx.add([_c(0, "The mitochondria is the powerhouse of the cell."),
             _c(1, "Cellular energy production occurs in specialized organelles.")])
    ranked = idx.rank("mitochondria")
    assert ranked[0][0] == 0            # exact lexical hit first
    assert all(s > 0 for _, s in ranked)

def test_rank_sorted_and_tiebroken_by_index():
    idx = BM25Index()
    idx.add([_c(0, "alpha beta"), _c(1, "alpha beta")])   # identical → same score
    ranked = idx.rank("alpha")
    assert [i for i, _ in ranked] == [0, 1]               # idx asc tiebreak

def test_empty_index_and_empty_query():
    assert BM25Index().rank("anything") == []
    idx = BM25Index(); idx.add([_c(0, "hello world")])
    assert idx.rank("") == []
    assert idx.rank("nonexistentterm") == []

def test_len_and_incremental_add():
    idx = BM25Index(); idx.add([_c(0, "a cat")]); idx.add([_c(1, "a dog")])
    assert len(idx) == 2
    assert idx.rank("dog")[0][0] == 1
```

- [ ] **Step 2:** `python -m pytest tests/test_lexical.py -q` → FAIL (no module).
- [ ] **Step 3:** Implement per spec §3.1 — postings `dict[str,list[(idx,tf)]]`, `df`, per-chunk length, running `N`/`avglen`; Okapi `idf = ln(1 + (N-df+0.5)/(df+0.5))`, score `Σ idf * (tf*(k1+1))/(tf + k1*(1-b + b*len/avglen))`; `rank` sorts `(-score, idx)`. Stdlib only (`re`, `math`).
- [ ] **Step 4:** `python -m pytest tests/test_lexical.py -q` → PASS. Then full `python -m pytest -q` green.
- [ ] **Step 5:** Commit `feat(retrieval): hand-built BM25 lexical index`.

---

### Task 2: Reciprocal Rank Fusion

**Files:** Create `doclens/fusion.py`; Test `tests/test_fusion.py`. Spec §3.2.

**Interfaces:**
- Produces: `rrf(rankings: list[list[int]], k_const: int = 60) -> list[tuple[int,float]]` — deduped, sorted score desc then idx asc; empty→[].

- [ ] **Step 1: Failing tests**

```python
from doclens.fusion import rrf

def test_item_high_in_both_wins():
    # idx 2 is rank1 in list A and rank1 in list B → highest fused
    out = rrf([[2, 0, 1], [2, 1, 0]])
    assert out[0][0] == 2

def test_single_list_preserves_order():
    assert [i for i, _ in rrf([[5, 3, 9]])] == [5, 3, 9]

def test_dedup_and_tiebreak_idx_asc():
    out = rrf([[0, 1], [1, 0]])          # symmetric → equal scores
    assert [i for i, _ in out] == [0, 1] # idx asc tiebreak
    assert len(out) == 2

def test_k_const_monotonicity():
    # larger k_const compresses rank gaps → rank-1 advantage shrinks
    small = dict(rrf([[0, 1]], k_const=1))
    big   = dict(rrf([[0, 1]], k_const=1000))
    assert (small[0] - small[1]) > (big[0] - big[1])

def test_empty():
    assert rrf([]) == [] and rrf([[]]) == []
```

- [ ] **Step 2:** pytest → FAIL. **Step 3:** implement (`score(idx)=Σ 1/(k_const+rank1based)`; accumulate in dict; sort `(-score, idx)`). **Step 4:** pytest green. **Step 5:** Commit `feat(retrieval): reciprocal rank fusion`.

---

### Task 3: Trace / Span / Tracer

**Files:** Create `doclens/trace.py`; Test `tests/test_trace.py`. Spec §3.5.

**Interfaces:**
- Produces: `Span(name,start_ms,end_ms,meta)` + `.duration_ms` + `.to_dict()`; `Trace(trace_id=None)` with `.trace_id` (12 hex), `.spans`, `.to_dicts()`, `.to_jsonl()`; `Tracer(trace=None)` with `@contextmanager span(name, **meta)` yielding the mutable `Span`.

- [ ] **Step 1: Failing tests**

```python
import json
from doclens.trace import Trace, Tracer

def test_span_records_duration_and_meta_mutation():
    t = Tracer()
    with t.span("retrieve", mode="hybrid") as sp:
        sp.meta["candidates"] = 20
    s = t.trace.spans[0]
    assert s.name == "retrieve" and s.duration_ms >= 0
    assert s.meta == {"mode": "hybrid", "candidates": 20}

def test_spans_appended_in_exit_order():
    t = Tracer()
    with t.span("a"): pass
    with t.span("b"): pass
    assert [s.name for s in t.trace.spans] == ["a", "b"]

def test_trace_id_is_12_hex():
    tid = Trace().trace_id
    assert len(tid) == 12 and all(c in "0123456789abcdef" for c in tid)

def test_to_dicts_and_jsonl_shape():
    t = Tracer()
    with t.span("embed", dims=768): pass
    d = t.trace.to_dicts()[0]
    assert {"name","start_ms","end_ms","duration_ms","meta"} <= set(d)
    assert json.loads(t.trace.to_jsonl().splitlines()[0])["name"] == "embed"
```

- [ ] **Step 2:** pytest → FAIL. **Step 3:** implement per spec §3.5 (`time.perf_counter()*1000`, `uuid.uuid4().hex[:12]`, `contextlib.contextmanager`). **Step 4:** green. **Step 5:** Commit `feat(trace): hand-built span/trace/tracer`.

---

### Task 4: `VectorIndex.rank_all` + `Retrieved.components`

**Files:** Modify `doclens/index.py`, `doclens/types.py`; Test `tests/test_index.py` (extend). Spec §4.1, §4.2.

**Interfaces:**
- Produces: `VectorIndex.rank_all(vector) -> list[tuple[int,float]]` (all chunks as (index, cosine), sorted score desc, index asc tiebreak; empty→[]); `Retrieved.components: dict = field(default_factory=dict)`.
- Consumes: existing `VectorIndex`, `Retrieved`.

- [ ] **Step 1: Failing tests** (extend `tests/test_index.py`)

```python
def test_rank_all_returns_all_sorted_with_index():
    from doclens.index import VectorIndex
    from doclens.types import Chunk
    idx = VectorIndex()
    ch = [Chunk(f"d-{i}", "d", 1, i, t) for i, t in enumerate(["a","b","c"])]
    idx.add(ch, [[1,0],[0,1],[1,1]])
    ranked = idx.rank_all([1,0])
    assert len(ranked) == 3
    assert ranked[0][0] == 0                       # exact match cosine highest
    assert [i for i,_ in ranked] == sorted(range(3), key=lambda i: (-dict(ranked)[i], i))

def test_rank_all_empty():
    from doclens.index import VectorIndex
    assert VectorIndex().rank_all([1,0]) == []

def test_retrieved_has_components_default():
    from doclens.types import Retrieved, Chunk
    r = Retrieved(chunk=Chunk("d-0","d",1,0,"x"), score=0.5)
    assert r.components == {}
```

- [ ] **Step 2:** pytest → FAIL. **Step 3:** add `rank_all` (normalize query, `scores=self._mat@q`, `order=np.argsort(-scores, kind="stable")`, return `[(int(i), float(scores[i])) for i in order]`; guard empty). Add `components` field to `Retrieved`. **Step 4:** green (existing index/types tests still pass). **Step 5:** Commit `feat(retrieval): VectorIndex.rank_all + Retrieved.components`.

---

### Task 5: HybridIndex

**Files:** Create `doclens/hybrid.py`; Test `tests/test_hybrid.py`. Spec §3.3.

**Interfaces:**
- Consumes: `VectorIndex.rank_all` (T4), `BM25Index.rank` (T1), `rrf` (T2), `Retrieved.components` (T4).
- Produces: `HybridIndex` — `add(chunks, vectors)`, `retrieve(qvec, qtext, *, mode="hybrid", pool=20) -> list[Retrieved]`, `__len__`, `.dense`, `.lexical`. `mode ∈ {dense,lexical,hybrid}`. `components` per candidate: `dense_score` (always), `dense_rank`, `bm25_rank` (None if absent), `rrf_score` (hybrid only).

- [ ] **Step 1: Failing tests**

```python
from doclens.hybrid import HybridIndex
from doclens.types import Chunk

def _mk():
    idx = HybridIndex()
    chunks = [Chunk(f"d-{i:04d}", "d", 1, i, t) for i, t in enumerate([
        "mitochondria powerhouse of the cell",     # 0 lexical+dense for mito
        "cellular energy organelles respiration",  # 1
        "the quick brown fox jumps",               # 2 unrelated
    ])]
    vecs = [[1,0,0],[0.6,0.4,0],[0,0,1]]
    idx.add(chunks, vecs)
    return idx

def test_dense_mode_matches_vectorindex_order():
    idx = _mk()
    got = [r.chunk.seq for r in idx.retrieve([1,0,0], "x", mode="dense", pool=3)]
    assert got == [r.chunk.seq for r in idx.dense.search([1,0,0], k=3)]

def test_components_populated_and_dense_score_always_present():
    idx = _mk()
    out = idx.retrieve([1,0,0], "mitochondria", mode="hybrid", pool=3)
    for r in out:
        assert "dense_score" in r.components          # every candidate
        assert "rrf_score" in r.components            # hybrid only
    # mitochondria: rank-1 in both dense and bm25 → fused first
    assert out[0].chunk.seq == 0

def test_bm25_rank_none_when_no_lexical_hit():
    idx = _mk()
    out = idx.retrieve([0,0,1], "fox", mode="hybrid", pool=3)
    top = next(r for r in out if r.chunk.seq == 2)
    assert top.components["bm25_rank"] is not None
    # a chunk with no query-term overlap has bm25_rank None
    assert any(r.components["bm25_rank"] is None for r in out)

def test_pool_caps_length_and_len():
    idx = _mk()
    assert len(idx) == 3
    assert len(idx.retrieve([1,0,0], "cell", mode="hybrid", pool=2)) == 2
```

- [ ] **Step 2:** pytest → FAIL. **Step 3:** implement per spec §3.3 (`dense_ranked=self.dense.rank_all(qvec)`; `cos_by_idx`, `dense_order`, `dense_rank`; branch on mode; build `Retrieved` with components; hybrid uses `rrf([dense_order, lex_order])`). **Step 4:** green + full suite. **Step 5:** Commit `feat(retrieval): HybridIndex (dense⊕BM25, RRF)`.

---

### Task 6: LLM listwise reranker

**Files:** Create `doclens/rerank.py`; Test `tests/test_rerank.py`. Spec §3.4.

**Interfaces:**
- Consumes: a `chat` with `.complete(messages, model) -> (text, Usage)`; `Retrieved` (+`components`).
- Produces: `llm_rerank(chat, model, question, candidates: list[Retrieved], top_k) -> tuple[list[Retrieved], Usage]`. Sets `components["rerank_rank"]` (1-based) on success; fallback returns `candidates[:top_k]` on any failure, no raise. `RERANK_INPUT_CHARS=400`.

- [ ] **Step 1: Failing tests** (fake chat, no network)

```python
from doclens.rerank import llm_rerank
from doclens.types import Retrieved, Chunk, Usage

def _cands(n):
    return [Retrieved(Chunk(f"d-{i:04d}","d",i+1,i,f"text {i}"), 0.5) for i in range(n)]

class FakeChat:
    def __init__(self, reply): self.reply, self.calls = reply, 0
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
```

- [ ] **Step 2:** pytest → FAIL. **Step 3:** implement per spec §3.4 — build listwise prompt (`text[:400]`), one `chat.complete([{ "role":"user","content":prompt }], model)`, parse first `[...]` via regex + `json.loads`, keep ints 1..N deduped, append missing in original order, map `n→candidates[n-1]`, slice `top_k`, set `rerank_rank`; wrap parse/call in try/except → `candidates[:top_k]`. **Step 4:** green. **Step 5:** Commit `feat(retrieval): LLM listwise reranker with safe fallback`.

---

### Task 7: `answer.py` — modes + rerank + tracer

**Files:** Modify `doclens/answer.py`; Test `tests/test_answer.py` (extend). Spec §4.3.

**Interfaces:**
- Consumes: `HybridIndex.retrieve` (T5), `llm_rerank` (T6), `Tracer` (T3).
- Produces: `answer_question(chat, chat_model, embedder, embed_model, index, question, k=5, history=None, *, retrieval_mode="hybrid_rerank", pool=20, tracer=None) -> AnswerResult`; module `RETRIEVAL_MODES=("dense","hybrid","hybrid_rerank")`. `index` is now a `HybridIndex`.

- [ ] **Step 1: Failing tests** (fake chat/embedder + HybridIndex)

```python
from doclens.answer import answer_question, RETRIEVAL_MODES
from doclens.hybrid import HybridIndex
from doclens.types import Chunk, Usage

class FakeEmb:
    def embed(self, texts, model): return [[1.0, 0.0] for _ in texts]
class FakeChat:
    def __init__(self, reply="Answer [p.1]"): self.reply, self.calls = reply, []
    def complete(self, messages, model):
        self.calls.append(messages); return self.reply, Usage(7, 3)

def _idx():
    idx = HybridIndex()
    idx.add([Chunk("d-0000","d",1,0,"alpha content"), Chunk("d-0001","d",2,1,"beta content")],
            [[1.0,0.0],[0.9,0.1]])
    return idx

def test_dense_mode_no_rerank_call_still_answers():
    chat = FakeChat()
    res = answer_question(chat, "m", FakeEmb(), "e", _idx(), "alpha", k=2,
                          retrieval_mode="dense")
    assert not res.refused and len(chat.calls) == 1     # generate only

def test_hybrid_rerank_invokes_reranker():
    chat = FakeChat("[1, 2]")   # first call = rerank JSON, second = answer... use two replies
    # simplest: assert 2 chat calls in hybrid_rerank
    class TwoReply:
        def __init__(s): s.n=0
        def complete(s, m, model):
            s.n+=1; return ("[2,1]" if s.n==1 else "Answer [p.1]"), Usage(5,2)
    c = TwoReply()
    res = answer_question(c, "m", FakeEmb(), "e", _idx(), "alpha", k=2,
                          retrieval_mode="hybrid_rerank")
    assert c.n == 2 and not res.refused
    assert res.usage.input_tokens == 10                 # 5 rerank + 5 generate

def test_low_cosine_refuses_in_every_mode():
    class OrthoEmb:
        def embed(self, texts, model): return [[0.0, 1.0] for _ in texts]  # ⟂ to docs → cos ~0
    for mode in RETRIEVAL_MODES:
        res = answer_question(FakeChat(), "m", OrthoEmb(), "e", _idx(), "zzz",
                              retrieval_mode=mode)
        assert res.refused, mode

def test_tracer_records_stage_spans():
    from doclens.trace import Tracer
    t = Tracer()
    answer_question(FakeChat(), "m", FakeEmb(), "e", _idx(), "alpha",
                    retrieval_mode="hybrid", tracer=t)
    names = [s.name for s in t.trace.spans]
    assert "embed" in names and "retrieve" in names and "generate" in names
    assert "rerank" not in names                        # hybrid (no rerank)
```

- [ ] **Step 2:** pytest → FAIL. **Step 3:** rewrite `answer_question` per spec §4.3 (tracer-or-`Tracer()`; embed/retrieve/[rerank]/generate spans; `base_mode`; refusal on `max(dense_score)`; `usage` accumulation). Keep `SYSTEM_PROMPT`/`REFUSAL_THRESHOLD`/`REFUSAL_TEXT`/`_CITE_RE`. **Step 4:** green (existing answer tests updated to build `HybridIndex`). **Step 5:** Commit `feat(answer): retrieval modes + rerank + tracing`.

**Note to implementer:** existing `tests/test_answer.py` builds a `VectorIndex`; migrate those to `HybridIndex` (same `add(chunks, vectors)` shape). This is expected, not scope creep.

---

### Task 8: `sessions.py` — HybridIndex + trace ring

**Files:** Modify `doclens/sessions.py`; Test `tests/test_sessions.py` (extend). Spec §4.4.

**Interfaces:**
- Consumes: `HybridIndex` (T5), `Trace` (T3).
- Produces: `SessionDoc.index: HybridIndex` (hint only); `SessionStore.add_trace(trace)`, `SessionStore.get_trace(trace_id) -> Trace | None`; `MAX_TRACES=200`, thread-safe (reuse existing lock), FIFO eviction.

- [ ] **Step 1: Failing tests**

```python
def test_add_and_get_trace():
    from doclens.sessions import SessionStore
    from doclens.trace import Trace
    s = SessionStore(); tr = Trace()
    s.add_trace(tr)
    assert s.get_trace(tr.trace_id) is tr
    assert s.get_trace("deadbeef0000") is None

def test_trace_ring_evicts_oldest():
    from doclens.sessions import SessionStore, MAX_TRACES
    from doclens.trace import Trace
    s = SessionStore(); first = Trace()
    s.add_trace(first)
    for _ in range(MAX_TRACES): s.add_trace(Trace())
    assert s.get_trace(first.trace_id) is None            # evicted past cap
```

- [ ] **Step 2:** pytest → FAIL. **Step 3:** add `_traces: OrderedDict[str,Trace]` guarded by the store lock; `add_trace` sets + pops oldest past `MAX_TRACES`; `get_trace` returns under lock. Update `SessionDoc.index` hint. **Step 4:** green. **Step 5:** Commit `feat(sessions): bounded trace ring`.

---

### Task 9: `server.py` — HybridIndex ingest, mode, trace event, export

**Files:** Modify `doclens/server.py`; Test `tests/test_server_ask.py` (extend), `tests/test_server_ingest.py` (spot-check). Spec §4.5.

**Interfaces:**
- Consumes: `answer_question(..., retrieval_mode, tracer)` (T7), `HybridIndex` (T5), `Tracer` (T3), `store.add_trace/get_trace` (T8), `RETRIEVAL_MODES` (T7).
- Produces: ingest builds `HybridIndex`; `/api/ask` accepts optional `mode` (validated ∈ `RETRIEVAL_MODES`, else default `"hybrid_rerank"`); emits `retrieval` chunks with `dense_rank`/`bm25_rank`/`rerank_rank`; emits `trace` event `{trace_id, spans}` before `answer`; `store.add_trace`. New `GET /api/trace/{trace_id}` → ndjson or 404.

- [ ] **Step 1: Failing tests** (TestClient, injected store; fake providers via existing patterns in the test module)

```python
# Assert on the SSE text of POST /api/ask (reuse the module's existing SSE harness):
# - a "trace" event line is present with a trace_id and a spans array
# - retrieval chunks include keys dense_rank / bm25_rank / rerank_rank
# - GET /api/trace/{that_id} returns 200 application/x-ndjson with >=1 line
# - GET /api/trace/unknownid returns 404
# - body {"mode":"bogus"} does not 500 (falls back to default and answers)
# - body {"mode":"dense"} answers without a rerank span in the trace
```
(Implementer: follow the existing `test_server_ask.py` fake-provider + SSE-parse helpers; add cases matching the bullets above.)

- [ ] **Step 2:** pytest → FAIL. **Step 3:** implement per spec §4.5. Preserve byo_key leak-safety (generic `except Exception` messages), `_DONE` on every path, `X-Accel-Buffering`. `/api/trace/{id}` returns `Response(trace.to_jsonl(), media_type="application/x-ndjson")` or 404 JSON. **Step 4:** green (full suite). **Step 5:** Commit `feat(server): retrieval mode, trace event, /api/trace export`.

---

### Task 10: Frontend — waterfall + fusion badges + persistence

**Files:** Modify `web/app.js`, `web/style.css`; (no `index.html` structural change). Spec §5.

**Interfaces:**
- Consumes: SSE `trace` event + enriched `retrieval` chunks (T9).
- Produces: `trace` SSE handler → `pendingTrace`; turn carries `trace`; `buildChunkCard` renders present rank badges; `buildTurnEl` renders `<details class="trace-details">` waterfall; `loadConvos` persists `trace`. **No new `innerHTML`.**

- [ ] **Step 1:** Manual acceptance criteria (no JS unit harness in repo — verify via a served smoke, matching how prior web tasks were validated):
  - After an ask, a "trace · N ms" disclosure appears with one bar per span; `generate` bar uses the accent fill.
  - Each source card shows the present badges (`dense #k`, `bm25 #k`, `rerank #k`); absent ranks omit their badge.
  - Reload the page → the waterfall persists on prior turns (from `convos.v1`).
  - `renderCitedAnswer` remains the only `innerHTML` assignment (grep `innerHTML` → 1 hit).
- [ ] **Step 2:** Implement per spec §5 with `createElement`/`textContent`/`style` only. Extend `loadConvos` turn map with `trace`. Add CSS classes `.trace-details`, `.span-row`, `.span-bar`, `.span-bar-fill`, `.rank-badge` in the ink+emerald palette.
- [ ] **Step 3:** Serve locally (`python -m uvicorn doclens.server:app`), ingest a small doc, ask, verify the four bullets; `grep -n innerHTML web/app.js` → exactly 1.
- [ ] **Step 4:** Commit `feat(web): trace waterfall + fusion badges`.

---

### Task 11: `evals/run.py` — mode axis

**Files:** Modify `evals/run.py`; Test `tests/test_eval_run.py` (extend). Spec §6.1.

**Interfaces:**
- Consumes: `answer_question(..., retrieval_mode=mode)` (T7), `HybridIndex` (T5).
- Produces: `run_eval(models, gold, out_path, *, modes=("dense","hybrid","hybrid_rerank"), embedder_factory, chat_factory, sleep_s)`; `--modes` CLI; records gain `"mode"`; resume key `(model, mode, case_id)`; `corpus_cache` builds `HybridIndex`.

- [ ] **Step 1: Failing tests** (fake factories, tiny gold — mirror existing test_eval_run patterns)

```python
# - run_eval over 1 model × 3 modes × 1 case → 3 records, each with distinct r["mode"]
# - resume: re-running with an existing (model,mode,case_id) in results adds no dup
# - corpus_cache builds HybridIndex (retrieve works; no VectorIndex attribute error)
```

- [ ] **Step 2:** pytest → FAIL. **Step 3:** add `modes` param + loop, `mode` in record, resume set keyed by triple, `HybridIndex` in `corpus_cache`, `retrieval_mode=mode` in the answer call, `--modes` arg. **Step 4:** green. **Step 5:** Commit `feat(evals): per-mode retrieval axis`.

---

### Task 12: `evals/report.py` — Mode column

**Files:** Modify `evals/report.py`; Test `tests/test_metrics.py` / report tests (extend). Spec §6.2.

**Interfaces:**
- Consumes: records with `mode` (T11).
- Produces: `summarize` groups by `(model, mode)` (adds `mode` field); `to_markdown` header `| Model | Mode | Recall@5 | MRR | Faithful | Refusal acc | p50 s |`, rows ordered model then `dense,hybrid,hybrid_rerank`; same splice markers.

- [ ] **Step 1: Failing tests**

```python
def test_summarize_groups_by_model_and_mode():
    from evals.report import summarize
    recs = [
      {"model":"m","mode":"dense","recall5":1.0,"mrr":0.5,"faithful":True,
       "refused_correctly":None,"latency_s":1.0},
      {"model":"m","mode":"hybrid","recall5":1.0,"mrr":0.8,"faithful":True,
       "refused_correctly":None,"latency_s":1.5},
    ]
    s = summarize(recs)
    assert {r["mode"] for r in s} == {"dense","hybrid"}

def test_markdown_has_mode_column_and_order():
    from evals.report import summarize, to_markdown
    recs = [{"model":"m","mode":m,"recall5":1.0,"mrr":0.5,"faithful":True,
             "refused_correctly":None,"latency_s":1.0}
            for m in ("hybrid_rerank","dense","hybrid")]
    md = to_markdown(summarize(recs))
    assert "| Mode |" in md
    assert md.index("dense") < md.index("hybrid_rerank")   # canonical order
```

- [ ] **Step 2:** pytest → FAIL. **Step 3:** implement per spec §6.2 (group key `(model,mode)`; mode sort rank `{"dense":0,"hybrid":1,"hybrid_rerank":2}`; add Mode column). **Step 4:** green. **Step 5:** Commit `feat(evals): report Mode column`.

---

### Task 13: Discriminating gold cases + probe corpus + README

**Files:** Modify `evals/gold.yaml`; maybe Create `evals/corpus/hybrid_probe.txt` (or `.md`/`.pdf` per `ingest_file`); Modify `README.md`. Spec §6.3.

**Interfaces:**
- Consumes: `fingerprint` (carry-forward constraints), `run_eval` modes (T11), report (T12).
- Produces: ≥4 gold cases whose relevant chunk is found by exact-term/rare-token signal; a methodology note + regenerated table in README.

- [ ] **Step 1:** Author the probe corpus (**English, distinct chunk openings** per carry-forward) with distinctive lexical anchors (exact error code / acronym / proper noun / exact figure among near-duplicate chunks). Add ≥4 gold cases (`id/doc/question/relevant_fps/expected_facts/answerable`), `relevant_fps` via `fingerprint`.
- [ ] **Step 2:** Run the eval across modes with a real key (`python -m evals.run --models gemini-3.1-flash-lite --modes dense,hybrid,hybrid_rerank --out evals/results.json`), then `python -m evals.report evals/results.json --readme README.md`.
- [ ] **Step 3: Acceptance —** MRR non-decreasing `dense ≤ hybrid ≤ hybrid_rerank`, with **at least one** mode strictly beating `dense`; faithful/refusal ≥ current. If flat, revise cases (that a flat delta is a real finding must be recorded in the ledger, not hidden). Add a README methodology paragraph: small corpus saturates recall@5; MRR/faithful carry the signal; the rerank latency tradeoff.
- [ ] **Step 4:** Commit `feat(evals): discriminating gold cases + per-mode README table`.

---

## Post-tasks
- Final whole-branch review (opus) over `git merge-base main HEAD..HEAD` with the Minor-findings roll-up.
- Live smoke on a real key (ingest → ask → waterfall renders → `/api/trace/{id}` exports).
- Then `superpowers:finishing-a-development-branch` (merge to main → Render redeploy).
