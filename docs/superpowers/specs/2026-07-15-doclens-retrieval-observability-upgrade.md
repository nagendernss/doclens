# doclens Retrieval + Observability Upgrade — Design Spec

**Status:** approved for planning (2026-07-15)

**Goal:** Upgrade doclens from single-stage dense retrieval to a two-stage
pipeline — hybrid (dense ⊕ BM25, RRF-fused) candidate retrieval followed by an
LLM listwise reranker — and add a hand-built tracing layer surfaced as a UI
waterfall, with the eval harness proving the before/after gain.

**Architecture:** A new `HybridIndex` wraps the existing dense `VectorIndex`
plus a hand-built `BM25Index`; their rankings are fused by Reciprocal Rank
Fusion. An LLM listwise reranker (one Gemini call) reorders the fused pool to
the final top-k. Each pipeline stage is wrapped in a timed `Span` collected
into a `Trace`, streamed to the browser as an SSE `trace` event and rendered as
a collapsible waterfall. The eval runner gains a `retrieval_mode` axis
(`dense` | `hybrid` | `hybrid_rerank`) so the README table reports each mode
side by side.

**Tech Stack:** Python (existing), numpy, httpx, FastAPI, PyYAML. Raw REST
Gemini adapters (no SDKs). Vanilla JS frontend (no build step, no CDN). **No
new dependencies** — BM25, RRF, the reranker, and tracing are all hand-built on
the stdlib (`re`, `math`, `uuid`, `contextlib`, `time`).

## Global Constraints

- **No new heavy deps. No torch, no sentence-transformers, no Langfuse SDK.**
  Everything must run on the current Render free-tier instance (512 MB) with no
  model downloads.
- **Raw httpx providers only** — the reranker reuses the existing `GeminiChat`
  adapter; no provider SDKs.
- **Refusal threshold `0.30` semantics unchanged** — refusal is decided on the
  **max dense cosine** among retrieved candidates, computed *before* rerank.
  BM25 and RRF scores are not calibrated to that threshold and must not feed it.
- **SSE contract is additive** — new `trace` event and new fields on existing
  `retrieval` chunks only. Old clients ignore unknown events/fields.
- **Frontend security invariant preserved** — `app.js` uses `innerHTML` at
  exactly one site (`renderCitedAnswer`). The waterfall and fusion badges are
  built with `textContent` / `createElement` / `style` only. No new `innerHTML`.
- **Secrets** — `byo_key` never logged or echoed; existing SSRF guard, rate
  limits, and session-reset-on-restart behavior untouched.
- **Defaults:** candidate pool `N = 20`, final `k = 5`, RRF `k_const = 60`,
  BM25 `k1 = 1.5` / `b = 0.75`, reranker candidate-text truncation `400` chars,
  `DEFAULT_RETRIEVAL_MODE = "hybrid_rerank"`, trace ring cap `200`.

---

## 1. Background — current pipeline

`answer_question(chat, chat_model, embedder, embed_model, index, question, k=5, history=None)`
(in `doclens/answer.py`) does:

1. Embed the question → `qvec`.
2. `index.search(qvec, k=5)` — cosine top-k over the in-memory `VectorIndex`.
3. Refuse if no hits or `retrieved[0].score < 0.30`.
4. Build `[p.N] …` context, prepend system + history, `chat.complete(...)`.
5. Parse `[p.N]` citations; return `AnswerResult`.

The dense index (`doclens/index.py`) is a normalized numpy matrix; the eval
harness (`evals/run.py`, `evals/metrics.py`, `evals/report.py`) already reports
recall@5 / MRR / faithful / refusal-accuracy / p50-latency and splices a table
into the README between `<!-- evals:start -->` / `<!-- evals:end -->`.

**Problem this upgrade addresses:** pure dense retrieval misses exact-term /
rare-keyword / proper-noun matches (embeddings blur them), and never reorders
by a stronger relevance signal. The current gold set is saturated
(recall@5 = 1.00), so it also can't *show* retrieval quality. This upgrade adds
the two standard fixes (hybrid + rerank), observability to see the stages, and
harder eval cases to measure the gain.

---

## 2. Target pipeline

```
question
  │  embed                      ── span: embed
  ▼
qvec
  │  hybrid retrieve (pool=20)  ── span: retrieve
  │    dense.rank_all ─┐
  │    bm25.rank ──────┴─ RRF fuse (k=60)
  ▼
candidates[≤20]  (each carries dense_score, dense_rank, bm25_rank, rrf_score)
  │  refusal gate: max(dense_score) < 0.30 → "Not in the document"
  │  LLM listwise rerank        ── span: rerank   (hybrid_rerank mode only)
  ▼
final[k=5]  (+ rerank_rank)
  │  generate                   ── span: generate
  ▼
answer + [p.N] citations   +   Trace{spans:[embed,retrieve,rerank,generate]}
```

`retrieval_mode` selects how far the pipeline runs:
- `dense` — baseline: `dense.rank_all` top-k, no fusion, no rerank.
- `hybrid` — RRF(dense, bm25) top-k, no rerank.
- `hybrid_rerank` — RRF pool → LLM rerank → top-k. **Server default.**

---

## 3. New modules

### 3.1 `doclens/lexical.py` — hand-built BM25

**Responsibility:** lexical ranking over the same chunks the dense index holds,
using the same 0..N-1 chunk-index space (chunks are added in one call, so
positions match the dense index exactly).

```python
STOPWORDS: frozenset[str]           # ~40 common English words
def _tokenize(text: str) -> list[str]
    # text.lower() → re.findall(r"[a-z0-9]+", …) → drop STOPWORDS and len<2 tokens

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None
    def __len__(self) -> int
    def add(self, chunks: list[Chunk]) -> None
        # tokenize each chunk.text; accumulate:
        #   self._postings: dict[str, list[tuple[int, int]]]  term -> [(chunk_idx, tf)]
        #   self._df: dict[str, int]                           term -> doc frequency
        #   self._len: list[int]                               chunk_idx -> token count
        #   self._n, self._avglen
    def rank(self, query: str) -> list[tuple[int, float]]
        # Okapi BM25 over query tokens; returns (chunk_idx, score) for every
        # chunk with score > 0, sorted score desc, chunk_idx asc as tiebreak.
        # idf = ln(1 + (N - df + 0.5) / (df + 0.5)); empty index or empty query → []
```

Incremental `add` (multiple calls append) mirrors `VectorIndex.add`. `avglen`
is recomputed from running totals on each `add`.

### 3.2 `doclens/fusion.py` — Reciprocal Rank Fusion

```python
def rrf(rankings: list[list[int]], k_const: int = 60) -> list[tuple[int, float]]
    # rankings: one best-first list of chunk indices per retriever.
    # score(idx) = Σ_r 1 / (k_const + rank_r)   with rank_r 1-based within list r
    #             (indices absent from a list contribute nothing from that list)
    # returns (idx, score) deduped, sorted score desc, idx asc as tiebreak.
    # empty input → []
```

RRF is chosen over weighted score blending because cosine (≈0–1) and BM25
(unbounded) live on different scales; rank fusion needs no per-corpus
normalization or tuning.

### 3.3 `doclens/hybrid.py` — HybridIndex

**Responsibility:** own the canonical chunk list and both retrievers; expose a
single `retrieve` that returns a candidate pool with full provenance in
`Retrieved.components`.

```python
class HybridIndex:
    def __init__(self) -> None
        self.dense = VectorIndex()
        self.lexical = BM25Index()
        self._chunks: list[Chunk] = []
    def __len__(self) -> int
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None
        # self.dense.add(chunks, vectors); self.lexical.add(chunks)
        # self._chunks.extend(chunks)   — index spaces stay aligned
    def retrieve(self, qvec: list[float], qtext: str, *,
                 mode: str = "hybrid", pool: int = 20) -> list[Retrieved]
```

`retrieve` logic:

```python
dense_ranked = self.dense.rank_all(qvec)          # [(idx, cosine)] desc  (NEW method, §4.1)
cos_by_idx   = dict(dense_ranked)
dense_order  = [i for i, _ in dense_ranked]
dense_rank   = {i: r for r, i in enumerate(dense_order, 1)}

if mode == "dense":
    chosen, primary = dense_order[:pool], cos_by_idx
    bm25_rank = {}
elif mode == "lexical":
    lex = self.lexical.rank(qtext)
    lex_order = [i for i, _ in lex]
    chosen, primary = lex_order[:pool], dict(lex)
    bm25_rank = {i: r for r, i in enumerate(lex_order, 1)}
else:  # hybrid
    lex = self.lexical.rank(qtext)
    lex_order = [i for i, _ in lex]
    bm25_rank = {i: r for r, i in enumerate(lex_order, 1)}
    fused = rrf([dense_order, lex_order])
    chosen, primary = [i for i, _ in fused][:pool], dict(fused)

out = []
for idx in chosen:
    comp = {"dense_score": cos_by_idx.get(idx, 0.0),
            "dense_rank": dense_rank.get(idx),
            "bm25_rank": bm25_rank.get(idx)}
    if mode == "hybrid":
        comp["rrf_score"] = primary.get(idx, 0.0)
    out.append(Retrieved(chunk=self._chunks[idx], score=primary.get(idx, 0.0),
                         components=comp))
return out
```

Invariant: `dense_score` is populated for **every** candidate regardless of
mode (so the refusal gate always has a calibrated cosine). `mode == "dense"`
returns exactly the same order/scores as today's `VectorIndex.search`.

### 3.4 `doclens/rerank.py` — LLM listwise reranker

```python
RERANK_INPUT_CHARS = 400
def llm_rerank(chat, model: str, question: str,
               candidates: list[Retrieved], top_k: int) -> tuple[list[Retrieved], Usage]
```

- `len(candidates) <= 1` → return `candidates[:top_k], Usage()` (no call).
- Build one listwise prompt:
  ```
  Rank the passages by how well they answer the question.
  Question: {question}

  [1] (p.3) {text[:400]}
  [2] (p.7) {text[:400]}
  ...
  Return ONLY a JSON array of the passage numbers, most relevant first,
  each number exactly once. Example: [3, 1, 2]
  ```
- `text, usage = chat.complete([{ "role":"user", "content": prompt }], model)`.
- Parse: first `[...]` substring → `json.loads` → keep ints in `1..N`, dedup
  preserving order, then append any missing candidate numbers in original order
  (guarantees ≥ `top_k` and total coverage). Map number → `candidates[n-1]`.
- Take `top_k`; set `components["rerank_rank"]` = 1-based position on each.
- **Fallback:** any exception / non-list / empty parse → return
  `candidates[:top_k], usage` (usage from the call if it happened, else
  `Usage()`); leave `rerank_rank` unset. Never raises, never drops below the
  available count.

### 3.5 `doclens/trace.py` — spans, trace, tracer

```python
@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float
    meta: dict = field(default_factory=dict)
    @property
    def duration_ms(self) -> float: return self.end_ms - self.start_ms
    def to_dict(self) -> dict:
        return {"name": self.name, "start_ms": round(self.start_ms, 3),
                "end_ms": round(self.end_ms, 3),
                "duration_ms": round(self.duration_ms, 3), "meta": self.meta}

class Trace:
    def __init__(self, trace_id: str | None = None) -> None
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.spans: list[Span] = []
    def to_dicts(self) -> list[dict]        # [s.to_dict() for s in self.spans]
    def to_jsonl(self) -> str               # one span JSON per line

class Tracer:
    def __init__(self, trace: Trace | None = None) -> None
        self.trace = trace or Trace()
    @contextmanager
    def span(self, name: str, **meta):
        start = time.perf_counter() * 1000
        sp = Span(name, start, start, dict(meta))
        try:
            yield sp                 # caller may mutate sp.meta after the call returns
        finally:
            sp.end_ms = time.perf_counter() * 1000
            self.trace.spans.append(sp)
```

`start_ms` values are relative (`perf_counter`); the UI normalizes by the
minimum span start for waterfall offsets. `uuid`/`time` are runtime-only (server
process) — unrelated to the workflow-script clock restriction.

---

## 4. Modified modules

### 4.1 `doclens/index.py`

Add one method (keeps `VectorIndex` otherwise pure):

```python
def rank_all(self, vector: list[float]) -> list[tuple[int, float]]:
    """All chunks as (index, cosine), sorted score desc, index asc as tiebreak.
    Empty index → []."""
```

`search()` can be re-expressed as `rank_all(vector)[:k]` mapped to `Retrieved`,
or left as-is (either is fine; no behavior change required).

### 4.2 `doclens/types.py`

Add provenance to `Retrieved` (additive, defaulted — all existing constructions
still valid):

```python
@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    components: dict = field(default_factory=dict)
```

### 4.3 `doclens/answer.py`

New signature (keyword-only additions, back-compatible call sites updated):

```python
RETRIEVAL_MODES = ("dense", "hybrid", "hybrid_rerank")

def answer_question(chat, chat_model, embedder, embed_model,
                    index: HybridIndex, question: str, k: int = 5,
                    history: list[dict] | None = None, *,
                    retrieval_mode: str = "hybrid_rerank", pool: int = 20,
                    tracer: Tracer | None = None) -> AnswerResult:
```

Flow (spans via `tracer = tracer or Tracer()` so a discarded trace is recorded
when none passed):

1. `embed` span → `qvec`; `sp.meta["dims"] = len(qvec)`.
2. `retrieve` span: `base_mode = "dense" if retrieval_mode == "dense" else "hybrid"`;
   `candidates = index.retrieve(qvec, question, mode=base_mode, pool=pool)`;
   `sp.meta.update({"mode": base_mode, "pool": pool, "candidates": len(candidates)})`.
3. Refusal: `top_dense = max((c.components.get("dense_score", c.score) for c in candidates), default=0.0)`.
   If `not candidates or top_dense < REFUSAL_THRESHOLD` → return refusal
   `AnswerResult` (retrieved = `candidates[:k]`, no rerank/generate).
4. If `retrieval_mode == "hybrid_rerank"`: `rerank` span →
   `final, ru = llm_rerank(chat, chat_model, question, candidates, top_k=k)`;
   `usage += ru`; `sp.meta.update({"in": len(candidates), "out": len(final), "input_tokens": ru.input_tokens, "output_tokens": ru.output_tokens})`.
   Else `final = candidates[:k]`.
5. `generate` span: build `[p.N]` context from `final`, system + history + user
   (unchanged prompt shape), `text, gu = chat.complete(...)`; `usage += gu`;
   record tokens in `sp.meta`.
6. Parse citations, compute `refused`, return `AnswerResult(... retrieved=final,
   usage=usage)`.

`SYSTEM_PROMPT`, `REFUSAL_THRESHOLD`, `REFUSAL_TEXT`, `_CITE_RE` unchanged.

### 4.4 `doclens/sessions.py`

- `SessionDoc.index` is now a `HybridIndex` (type hint only; `.index` still
  passed straight to `answer_question`).
- Add a bounded trace ring to `SessionStore`:
  ```python
  MAX_TRACES = 200
  def add_trace(self, trace: Trace) -> None      # OrderedDict keyed by trace_id, evict oldest past cap
  def get_trace(self, trace_id: str) -> Trace | None
  ```
  Keyed by the unguessable `trace_id` (uuid4) so cross-session enumeration isn't
  practical; global cap bounds memory. (Trade-off noted: not sid-scoped, which
  keeps the export endpoint stateless-simple; acceptable for an ephemeral demo
  where traces vanish on restart anyway.)

### 4.5 `doclens/server.py`

- **Ingest:** build `HybridIndex()` instead of `VectorIndex()` (BM25 is built
  inside `HybridIndex.add`, no extra embed cost). Progress stages unchanged.
- **Ask:** parse optional `mode` from the request body, validated against
  `RETRIEVAL_MODES`, default `DEFAULT_RETRIEVAL_MODE = "hybrid_rerank"`. Create
  `tracer = Tracer()`; pass `retrieval_mode=mode, tracer=tracer` into
  `answer_question`. After the answer:
  - `retrieval` event chunks gain provenance fields:
    ```python
    {"page": r.chunk.page, "score": r.score,
     "preview": r.chunk.text[:RETRIEVAL_PREVIEW_CHARS],
     "dense_rank": r.components.get("dense_rank"),
     "bm25_rank":  r.components.get("bm25_rank"),
     "rerank_rank": r.components.get("rerank_rank")}
    ```
  - Emit a new `trace` event **before** `answer`:
    `emit("trace", {"trace_id": tracer.trace.trace_id, "spans": tracer.trace.to_dicts()})`,
    then `store.add_trace(tracer.trace)`.
- **New endpoint** `GET /api/trace/{trace_id}` → `store.get_trace`; if found,
  `Response(trace.to_jsonl(), media_type="application/x-ndjson")`; else 404 JSON
  `{"error": "trace not found"}`. Wrapped in the existing error-hygiene style.

### 4.6 `doclens/providers/registry.py`

No functional change required — the reranker reuses `get_chat`. The
`CHAT_MODELS` price columns (currently `0.0`) remain the source for any future
cost-per-span; tracing records raw tokens now, cost stays `0.0` until priced
models are added. (No code change unless we later want per-span cost — out of
scope here.)

---

## 5. Frontend — waterfall + fusion badges (`web/app.js`, `web/style.css`)

- **SSE handling** in `handleAskSubmit`: add a `trace` handler storing
  `pendingTrace = { trace_id, spans }`; include it on the appended turn:
  `trace: pendingTrace`.
- **`buildChunkCard(c)`**: after the score row, append a small badge row (only
  for fields that are present) built with `createElement`/`textContent`:
  `dense #{dense_rank}` · `bm25 #{bm25_rank}` · `rerank #{rerank_rank}`. Missing
  (`null`) ranks are omitted (e.g. a chunk BM25 never scored shows no bm25
  badge). No `innerHTML`.
- **`buildTurnEl(turn)`**: if `turn.trace?.spans?.length`, append a
  `<details class="trace-details">` with summary `trace · {totalMs} ms` and one
  `.span-row` per span:
  - label (`embed` / `retrieve` / `rerank` / `generate`),
  - a `.span-bar` whose `.span-bar-fill` width = `duration_ms / maxDuration`
    (proportional; `style.width` set numerically),
  - `{duration_ms} ms` and, when present, `{input+output} tok`.
  Built entirely with DOM APIs.
- **Persistence:** extend the `convos.v1` turn map in `loadConvos` to carry
  `trace: (t.trace && typeof t.trace === "object") ? t.trace : null` (additive;
  old records without it render no waterfall). Bump the inline schema comment.
- **`style.css`:** add `.trace-details`, `.span-row`, `.span-bar`,
  `.span-bar-fill`, `.rank-badge` in the existing ink+emerald palette. The
  `generate` span uses the accent fill; retrieval stages a muted fill so the
  dominant cost reads at a glance.

No changes to `index.html` structure (turns are built dynamically).

---

## 6. Eval — prove the gain

### 6.1 `evals/run.py`

- Add a `modes` axis (default `("dense", "hybrid", "hybrid_rerank")`) and
  `--modes dense,hybrid,hybrid_rerank` CLI flag.
- `resume_set` key becomes `(model, mode, case_id)`; each record gains
  `"mode": mode`.
- `corpus_cache` builds a `HybridIndex` per doc (embed once; BM25 built in
  `add`). The answer call becomes
  `answer_question(chat, chat_model, embedder, embed_model, index, question, k=5, retrieval_mode=mode)`.
- `latency_s` now naturally includes the rerank call for `hybrid_rerank` — that
  is the tradeoff the table is meant to show (p50 latency per mode).

### 6.2 `evals/report.py`

- `summarize` groups by `(model, mode)` and emits a `mode` field per row.
- `to_markdown` gains a **Mode** column:
  `| Model | Mode | Recall@5 | MRR | Faithful | Refusal acc | p50 s |`,
  rows ordered by model then mode (`dense`, `hybrid`, `hybrid_rerank`).
- Same `<!-- evals:start/end -->` splice markers.

### 6.3 Harder gold cases (`evals/gold.yaml` + corpus)

The current gold set saturates dense retrieval (recall@5 = 1.00), leaving no
measurable delta. Add cases whose relevant chunk is found by *exact-term /
rare-token* signals that embeddings blur:

- exact error code / identifier (e.g. a specific code, SKU, or version string),
- a proper noun / acronym that appears verbatim in one chunk,
- a numeric/date lookup where several chunks are semantically similar but only
  one has the exact figure,
- a multi-term query where lexical overlap disambiguates near-duplicate chunks.

Author these against the existing corpus where such terms already occur; if the
existing corpus lacks strong lexical probes, add one small doc
`evals/corpus/hybrid_probe.pdf` (or `.txt`, matching the ingest path used by
`ingest_file`) containing distinctive terms. `relevant_fps` are produced with
the existing `fingerprint(doc_id, page, text)` helper, same as current cases.

**Acceptance for this section:** on the expanded gold set, MRR is
non-decreasing from `dense` → `hybrid` → `hybrid_rerank`, with at least one
mode showing a strictly higher MRR than `dense` (i.e. the harder cases actually
exercise the new stages). Faithful/refusal stay ≥ current numbers. If the delta
doesn't appear, the gold cases aren't discriminating and need revision — that
is a real finding, to be surfaced, not hidden.

---

## 7. Testing strategy

New test files mirror the module layout; extend existing ones where noted. All
provider calls are faked (monkeypatched `chat`/`embedder`) as in the current
`test_answer.py` / `test_providers.py`.

- **`tests/test_lexical.py`** — exact-term chunk outranks a lexically-different
  paraphrase; idf lowers common terms; stopwords/short tokens dropped; empty
  index and empty query → `[]`; incremental `add`.
- **`tests/test_fusion.py`** — an item ranked high in both lists wins; single
  list preserves order; disjoint lists interleave by reciprocal rank; dedup;
  `k_const` monotonicity; empty → `[]`.
- **`tests/test_hybrid.py`** — `dense` mode equals `VectorIndex` order;
  `hybrid` fuses; `components` populated (`dense_score` on every candidate;
  ranks correct; `rrf_score` only in hybrid); `pool` caps length; `__len__`.
- **`tests/test_rerank.py`** — valid JSON reorders; garbage/empty → original
  `top_k` (fallback) with no raise; invalid/duplicate IDs filtered; missing IDs
  appended; `<=1` candidate makes no call; `Usage` propagated; `rerank_rank`
  set on success, absent on fallback.
- **`tests/test_trace.py`** — `duration_ms >= 0`; spans appended in exit order;
  `meta` mutation after the call is captured; `to_dicts`/`to_jsonl` shape;
  `trace_id` present and 12 hex chars.
- **`tests/test_answer.py`** (extend) — `dense` mode reproduces baseline
  retrieval; `hybrid_rerank` invokes the reranker (spy); low-cosine question
  refuses in **every** mode; `usage` sums rerank + generate; tracer records
  `embed`/`retrieve`/`generate` always and `rerank` only in rerank mode.
- **`tests/test_server_ask.py`** (extend) — `trace` SSE event emitted with
  `trace_id` + spans; `retrieval` chunks carry `dense_rank`/`bm25_rank`/
  `rerank_rank`; `GET /api/trace/{id}` returns ndjson for a known id and 404 for
  unknown; body `mode` validated (bad mode → default, not a 500).
- **`tests/test_sessions.py`** (extend) — `add_trace`/`get_trace`; eviction past
  `MAX_TRACES`.
- **`tests/test_eval_run.py`** (extend) — records carry `mode`; resume keyed by
  `(model, mode, case_id)`; all modes iterated.
- **`tests/test_metrics.py` / report** (extend) — `summarize` groups by
  `(model, mode)`; markdown has the Mode column and correct row order.

---

## 8. Out of scope (YAGNI)

- Persistent / on-disk vector store (retrieval stays in-memory per session).
- Cross-encoder reranker / torch / sentence-transformers.
- Langfuse or OpenTelemetry export (hand-built trace only).
- Query rewriting / HyDE / multi-query (a possible *future* card).
- ColBERT / multi-vector retrieval; reranker fine-tuning.
- Token-by-token answer streaming (answer still arrives as one SSE `answer`).

---

## 9. File summary

**New:** `doclens/lexical.py`, `doclens/fusion.py`, `doclens/hybrid.py`,
`doclens/rerank.py`, `doclens/trace.py`; tests `tests/test_lexical.py`,
`tests/test_fusion.py`, `tests/test_hybrid.py`, `tests/test_rerank.py`,
`tests/test_trace.py`; possibly `evals/corpus/hybrid_probe.*`.

**Modified:** `doclens/index.py` (+`rank_all`), `doclens/types.py`
(+`Retrieved.components`), `doclens/answer.py` (modes + rerank + tracer),
`doclens/sessions.py` (HybridIndex + trace ring), `doclens/server.py`
(HybridIndex ingest, `mode`, `trace` event, `/api/trace/{id}`), `web/app.js`
(trace handler, badges, waterfall, persistence), `web/style.css` (waterfall +
badges), `evals/run.py` (modes), `evals/report.py` (Mode column),
`evals/gold.yaml` (harder cases), `README.md` (table columns; brief
architecture note).
