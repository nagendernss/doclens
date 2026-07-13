# doclens — Design Spec

**Date:** 2026-07-14 · **Owner:** Nagender Swaroop Srivastava · **Status:** Approved

## 1. Summary

doclens is a deployable RAG web app: upload a PDF (≤10 MB) or paste a URL
(webpage or PDF link), the document is parsed → chunked → embedded into an
in-memory vector index, then questions are answered strictly from retrieved
chunks with page-level citations. Ships with an eval harness measuring
retrieval quality (recall@5, MRR) and answer faithfulness, published in the
README. Sibling project to repolens (same provider-adapter and eval-first
philosophy; the two link to each other).

**Portfolio goal:** make "RAG, embeddings, vector search, retrieval evals"
honestly claimable.

## 2. Goals / Non-goals

Goals:
- Public demo strangers use with zero setup at $0/month baseline (Render free).
- Real ingestion engineering: PDF text+page extraction, HTML cleaning,
  overlapping chunking with metadata.
- Hand-built cosine-similarity vector index (numpy) — pluggable store
  interface with a documented Qdrant scale path.
- Provider-agnostic embedding + chat adapters on raw httpx (Gemini free tier
  first; OpenAI slots) — same pattern as repolens.
- Grounded answering: context-only prompt, `[p.N]` citations, explicit
  "not in the document" refusals.
- Reproducible eval harness: pinned seed corpus, gold labels, deterministic
  metrics, README table generated from real runs.

Non-goals (v1): multi-page crawling, OCR/scanned PDFs, accounts/persistence,
rerankers, hybrid BM25 (future-work section), conversation memory.

## 3. Architecture

Single FastAPI service, vanilla-JS frontend, SSE streaming — the repolens
shape. New core packages:

```
doclens/
  ingest.py      # bytes/URL → Document(pages/blocks)  [pypdf, selectolax]
  chunker.py     # Document → list[Chunk] (≈500-token windows, 15% overlap,
                 #   metadata: page, heading, seq)
  embeddings/    # EmbeddingProvider protocol; gemini.py (REST, free tier),
                 #   openai slot; batch + retry/backoff
  index.py       # VectorIndex: add(chunks, vectors), search(vector, k) →
                 #   [(chunk, score)]; numpy cosine; Store protocol with
                 #   InMemoryStore now, qdrant documented later
  answer.py      # question → embed → top-k → grounded prompt → chat provider
                 #   (adapters ported from repolens) → AnswerResult(text,
                 #   citations[chunk_id/page], scores, usage)
  sessions.py    # per-visitor corpus: cookie sid → docs+index, 30 min idle
                 #   TTL, caps (3 docs, ~1500 chunks), thread-safe sweeper
  server.py      # POST /api/ingest (multipart file | {"url"}) → SSE progress;
                 #   POST /api/ask → SSE retrieval+answer; /api/models,
                 #   /healthz, static
  ratelimit.py   # ported from repolens (per-IP + global daily, BYO bypass)
  cli.py         # doclens ingest <path|url> + doclens ask "<q>" (local dev)
evals/           # seed corpus (3 pinned public docs in repo), gold.yaml
                 #   (~30 q → relevant chunk labels + fact regexes), runner,
                 #   metrics: recall@5, MRR, faithfulness; README splicer
web/             # index.html, app.js, style.css (doclens-branded, same family)
```

SSE contracts:
- ingest: `progress` {"stage": "download|parse|chunk|embed", "done", "total"} →
  `ready` {"doc_id", "title", "pages", "chunks"} | `error` {"message"}
- ask: `retrieval` {"chunks": [{"id", "page", "score", "preview"}]} →
  `answer` {"answer", "citations": [{"chunk_id", "page"}], "model",
  "input_tokens", "output_tokens"} | `error` {"message"}

## 4. Key behaviors & caps

- Upload: PDF only v1, ≤10 MB, ≤300 pages. URL: single page; content-type
  text/html → clean text (selectolax, keep title/h1-h3 structure); PDF →
  same PDF path; ≤5 MB fetched, 15 s timeout, no redirects off-scheme,
  private-network hosts refused (SSRF guard: resolve + reject non-public IPs).
- Chunking: target ~500 tokens (approx 4 chars/token heuristic), 15% overlap,
  never split mid-sentence when a boundary exists within ±80 chars.
- Retrieval: top-5 default; answer prompt includes chunk texts with `[p.N]`
  tags; model instructed to cite per-claim and refuse when context is
  insufficient (max score < threshold → short-circuit "not in the document"
  without an LLM call).
- Sessions: cookie `dl_sid` (random, httponly); idle TTL 30 min; caps
  3 docs / ~1500 chunks / session; background sweeper thread; Render restart
  wipes everything — stated in UI.
- Rate caps: `PER_IP_DAILY_CAP` (default: 3 ingests, 15 questions),
  `DAILY_GLOBAL_CAP`; BYO Gemini key bypasses and is never stored/logged.
- Embedding calls batched (≤64 chunks/request), retry/backoff on 429/5xx,
  eval runner sequential.

## 5. Eval harness

- Seed corpus pinned IN the repo (3 small public-domain/CC docs: one academic
  paper PDF, one technical manual excerpt, one policy-style doc).
- `evals/gold.yaml`: ~30 cases — `id`, `doc`, `question`,
  `relevant_chunks` (label by stable chunk fingerprint: doc + page +
  normalized 8-word prefix, robust to re-chunking), `expected_facts`
  (regex, all-of), `answerable: true|false` (5+ unanswerable cases must
  produce refusals).
- Metrics: recall@5 (any relevant chunk retrieved), MRR, faithfulness =
  fact regexes pass AND every citation points at a retrieved chunk,
  refusal-accuracy for unanswerable cases. Cost/latency per model.
- `python -m evals.run --models ... --out results.json` (resumable, atomic
  writes) → `python -m evals.report results.json --readme README.md`.

## 6. Deployment

Docker (python:3.12-slim) → Render free blueprint (`render.yaml`), health
`/healthz`, env: `GEMINI_API_KEY`, caps. In-memory everything; no disk.

## 7. Testing

pytest, no live network: fixture PDFs (tiny, generated in-repo), recorded
HTML fixtures, MockTransport for embedding/chat adapters, TestClient for SSE.
Index math tested against hand-computed cosine values. Session TTL via
injectable clock. Target: same rigor as repolens (~100 tests).

## 8. Milestones

1. Core: ingest (PDF+URL) → chunker → embeddings adapter → index → answer,
   CLI end-to-end (Plan A).
2. Evals: seed corpus, gold set, metrics, README table (Plan A).
3. Server/SSE + sessions + rate caps + frontend (Plan B).
4. Docker/Render deploy + live verify + cross-link repolens/portfolio (Plan B).

## 9. Risks

- Free-tier embedding quotas → batch + backoff; eval runner resumable;
  document daily limits in README.
- PDF text extraction variance → seed corpus pinned; extraction quirks noted;
  OCR out of scope.
- SSRF via URL ingestion → IP-resolution guard, scheme allowlist, size cap.
- Memory growth → session caps + TTL sweeper; documented restart semantics.
