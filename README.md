# doclens

Upload a PDF or paste a link. Ask questions. Get answers cited to the page — with retrieval
quality measured, not vibed.

[![ci](https://github.com/nagendernss/doclens/actions/workflows/ci.yml/badge.svg)](https://github.com/nagendernss/doclens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Quickstart

```bash
pip install -e .
export GEMINI_API_KEY=...        # free: https://aistudio.google.com/apikey
doclens ask paper.pdf "what's the main result?"
```

Windows PowerShell: `$env:GEMINI_API_KEY="..."` · Windows cmd: `set GEMINI_API_KEY=...` instead
of `export`.

```
doclens ask <path-or-url> "<question>" [--model gemini-3.1-flash-lite] [-k 5]
doclens models                                    # list models, * = key present
```

A URL works the same way as a local file — PDF link or webpage, both get the same SSRF-guarded
fetch path:

```bash
doclens ask https://example.com/some-article "what does it say about X?"
```

## How it works

Deep dive: [design spec](docs/superpowers/specs/2026-07-14-doclens-design.md) ·
[task-by-task implementation plan](docs/superpowers/plans/2026-07-14-doclens-core.md).

```mermaid
flowchart LR
    A[PDF or URL] --> B[ingest]
    B --> C[chunk]
    C --> D[embed]
    D --> E[(vector index)]
    Q[question] --> F[embed question]
    F --> E
    E -->|top score below 0.30| R[refuse — no LLM call]
    E -->|top-5 cosine matches| G[answer + p.N citations]
```

| Stage | Module | What it does |
|---|---|---|
| Ingest | `doclens/ingest.py`, `doclens/ingest_url.py` | PDF (`pypdf`) or URL (`httpx` + `selectolax`) → a `Document` of per-page text; SSRF guard on every URL |
| Chunk | `doclens/chunker.py` | Sliding window, ~2,000 chars (~500 tokens) per chunk, 15% overlap, snapped back to a sentence boundary up to 80 chars earlier |
| Embed | `doclens/providers/gemini.py` | Batches of ≤64 chunks → `gemini-embedding-001` vectors, with 429/5xx retry |
| Index | `doclens/index.py` | Hand-built numpy `VectorIndex` — L2-normalize rows, dot product = cosine, top-k |
| Answer | `doclens/answer.py` | Embed the question, retrieve top-5, build a context-only prompt, cite `[p.N]` per claim, refuse below the score threshold |

The answer prompt is deliberately narrow: "use only the context chunks below, cite every claim as
`[p.N]`, and if the context doesn't contain the answer, reply starting with exactly `Not in the
document.`" Citations are extracted from the model's own output with a regex
(`\[p\.(\d+)\]`), not trusted blindly — the eval harness later checks every citation actually
points at a chunk that was retrieved.

Shape of a run (illustrative — see [Evals](#evals) for real numbers; this is not a captured live
transcript):

```
$ doclens ask evals/corpus/device-manual.md "What voltage and wattage does the BX-9 require?"
doclens · device-manual.md · 3 pages · 6 chunks
retrieved: p.1, p.1, p.2, p.2, p.3

The Boreal SmartBrew BX-9 requires 120V at 60Hz and draws 1350W [p.1]. Do not operate it on any
other voltage or frequency [p.1].

model=gemini-3.1-flash-lite tokens=147+31
```

## Evals

<!-- evals:start -->
| Model | Recall@5 | MRR | Faithful | Refusal acc | p50 s |
|-------|----------|-----|----------|-------------|-------|
| _(not yet run)_ | — | — | — | — | — |
<!-- evals:end -->

Populate the table above with a real run:

```bash
python -m evals.run --models gemini-3.1-flash-lite --out results.json
python -m evals.report results.json --readme README.md
```

**Methodology.** 30 gold cases (24 answerable + 6 unanswerable, 8+2 per document) against 3
original documents in `evals/corpus/`, each chunking to 6–7 pieces (19 chunks total across the
corpus — see [Design decisions](#design-decisions)). Gold labels are chunk **fingerprints**
(`doc_id|p<page>|<first 8 normalized words>`), not chunk IDs, so re-chunking the corpus doesn't
silently invalidate the gold set. Grading is fully deterministic — no LLM judge:

- **Recall@5** — 1.0 if any gold-relevant chunk fingerprint is in the top-5 retrieved, else 0.0.
- **MRR** — 1 / rank of the first relevant chunk (0 if none in the top-5).
- **Faithful** — every `expected_facts` regex matches the answer text, AND every `[p.N]` citation
  in the answer points at a page that was actually retrieved.
- **Refusal accuracy** — fraction of the 6 unanswerable cases the pipeline correctly refused
  ("Not in the document"), ideally via the 0.30 cosine short-circuit rather than an LLM call.

**Honest caveat, stated on purpose:** with only 6–7 chunks per document and `k=5`, most of a
document's chunks come back on nearly every question — recall@5 saturates near 1.0 regardless of
retrieval quality, so on this seed corpus it isn't a discriminating metric. **MRR** (does the
*most* relevant chunk rank first, not just land in the top 5) and **faithfulness / refusal
accuracy** (does the model actually ground itself in what it retrieved, and stay quiet when it
shouldn't answer) carry the real retrieval-quality signal here. A meaningfully larger corpus
would be needed before recall@5 says anything a coin flip couldn't.

The runner (`evals/run.py`) ingests, chunks and embeds each corpus document once — not once per
question — caching and reusing that pass across every model and case. It's resumable:
`(model, case_id)` pairs already present in `results.json` are skipped on the next run, and every
record is written through a `.tmp` file + `os.replace` swap, so a rate-limit pause or a crash
mid-run never corrupts progress. `evals/report.py` turns records into the table above and splices
it between the `<!-- evals:start -->` / `<!-- evals:end -->` markers, replacing whatever was
there before — including this placeholder row.

## Design decisions

### Hand-built cosine index, not FAISS/Chroma/Qdrant

At this corpus scale (tens to low thousands of chunks, single process, in-memory) an
approximate-nearest-neighbor library buys nothing — brute-force cosine search over a numpy matrix
is exact, sub-millisecond at this size, and adds zero dependencies to pin, build, or explain.
`doclens/index.py`'s `VectorIndex` is deliberately three methods (`add`, `search`, `__len__`):
rows are L2-normalized on insert (a zero vector stays zero instead of dividing by zero into NaN —
`test_zero_vector_safe` pins that), search is `normalized_matrix @ query`, and ties are broken
with a stable sort so results are deterministic run to run (`np.argsort(-scores, kind="stable")`).
Swapping in Qdrant or pgvector at real scale means replacing this one class behind the same narrow
surface — not a `Protocol` formally declared in code today, but the interface is intentionally
already that small.

### Fingerprint-based gold labels, not chunk IDs

`Chunk.chunk_id` (`{doc_id}-{seq:04d}`) shifts the moment chunking parameters change — a bigger
target size, a different overlap, a chunker bugfix all reshuffle sequence numbers. Gold labels
reference `types.fingerprint(doc_id, page, text)` instead: the page number plus the first 8
normalized words of the chunk. Re-chunk the corpus and most fingerprints still resolve, because
they're keyed to *content*, not to an index assigned at chunk time. The gold set's fingerprint
references were never hand-typed — a throwaway script ran the real `ingest_file → chunk_document →
fingerprint` pipeline over the committed corpus and its output was copied verbatim into
`evals/gold.yaml`. That process caught a genuine quirk in the normalizer before it became a
mislabeled case: it strips newlines without inserting a space, so text spanning a paragraph break
can glue together — one more reason to generate fingerprints from real code, not by hand.

### Refusal threshold — a cosine cutoff, not a vibe

`REFUSAL_THRESHOLD = 0.30` in `answer.py`: if the top retrieved chunk's cosine score is below
0.30 (or nothing was retrieved), `answer_question` returns "Not in the document" **without
calling the chat model at all** — `test_low_score_refuses_without_llm` asserts the fake chat
provider is never invoked on that path. Short-circuiting beats always asking the model to judge
its own relevance for two reasons: it's free (no tokens spent asking a model to tell you it
doesn't know), and it's deterministic (a threshold on retrieval scores can't be talked out of an
answer the way a model merely instructed to "refuse if unsure" sometimes can). The prompt also
carries a second, independent refusal path — the model itself must open with "Not in the
document." when the retrieved context doesn't cover the question — so a topically-adjacent but
wrong retrieval still has a chance to be caught. Refusal accuracy in the eval harness grades both
paths together across the 6 unanswerable gold cases.

### SSRF guard — hardened past the basic private-IP check

The first pass rejected `is_private`/loopback/link-local IPs on the resolved hostname — the
obvious check, and the one most guides stop at. Follow-up hardening closed three real gaps,
covered by 16 tests in `tests/test_ingest_url.py`:

- **CGNAT.** `is_private` alone misses `100.64.0.0/10` (RFC 6598) — carrier-grade NAT space ISPs
  use to front many customers behind one public IP, which can still route to carrier-internal
  hosts. The guard now gates on `ipaddress.is_global` first, keeping the explicit loopback /
  link-local / reserved / multicast / unspecified denies as defense in depth.
- **DNS rebinding (TOCTOU).** Validating a hostname and then letting the HTTP client re-resolve it
  moments later at connect time is two independent lookups — an attacker controlling DNS can
  answer them differently. `_assert_public` and the actual connection now share one resolution:
  the validated IP is pinned straight into the request (`_pin_to_ip`), with the original `Host`
  header preserved.
- **Redirect chains.** A URL that redirects through a public host into a private one previously
  only had its *final* URL checked. Every hop is now re-validated before it's contacted (max 5
  hops), and `test_redirect_bounceback_blocked` / `test_redirect_to_private_blocked` confirm the
  private host is never actually requested — not just that the end result errors.
- **Streamed size cap.** The 5 MB URL cap aborts mid-download (`resp.iter_bytes()`) instead of
  buffering an oversized body fully into memory first and rejecting it afterward —
  `test_stream_size_cap` asserts the cap fires before all chunks are pulled.

Known residual gaps, accepted and documented rather than silently left open: 6to4/site-local IPv6
ranges aren't specifically enumerated (caught by `is_global` in practice, not by an explicit
deny), and pinning an `https://` URL to a validated IP repoints TLS SNI at the IP literal, which
fails certificate verification against a real HTTPS server rather than silently downgrading
security — a fully correct fix needs a custom transport that pins the socket while still
presenting the original hostname for SNI.

### Original authored eval corpus

All three seed documents (`evals/corpus/*.md`) are original prose written for this project — a
fictional network protocol spec, a fictional coffee-maker manual, a fictional retail leave/refund
policy — not scraped or excerpted from a real source. That means zero copyright/licensing risk, a
corpus that can't change out from under the gold labels the way a live webpage can, and facts that
were controllable at authoring time: numbers and edge cases were written to be regex-checkable and
unique to their own document, so there's no cross-document term overlap that could produce a
spurious retrieval match. Every corpus doc is plain ASCII on purpose — `types.fingerprint()`
normalizes to `[a-z0-9 ]` only, and non-ASCII punctuation (em/en dashes, curly quotes) collapses
in ways that make fingerprints harder to verify by eye.

### Raw httpx providers, no SDKs

Same rule as [repolens](https://github.com/nagendernss/repolens): no LangChain, no
`google-generativeai`. `doclens/providers/gemini.py` translates the pipeline's plain-dict message
and embedding calls into Gemini's REST shapes (`generateContent`, `batchEmbedContents`) over
`httpx`, and `providers/_http.py` is one shared `post_with_retry` — 429/5xx back off 2s → 4s → 8s,
4xx fails fast — used by both provider methods and the eval runner. Five runtime dependencies
total (`httpx`, `pyyaml`, `pypdf`, `selectolax`, `numpy`): one HTTP client, not a provider SDK
with its own object model and release schedule to track.

## Caps

| Cap | Value | Enforced in |
|---|---|---|
| PDF upload size | 10 MB | `ingest.py` — `MAX_PDF_BYTES` |
| PDF page count | 300 pages | `ingest.py` — `MAX_PDF_PAGES` |
| URL fetch size | 5 MB, streamed (aborts mid-download over cap) | `ingest_url.py` — `MAX_URL_BYTES` |
| URL fetch timeout | 15 s | `ingest_url.py` — `TIMEOUT_S` |
| URL redirect hops | 5, each hop re-validated against the SSRF guard before it's contacted | `ingest_url.py` — `MAX_REDIRECTS` |
| Chunk size target | 2,000 chars (~500 tokens), snapped back to a sentence boundary up to 80 chars earlier | `chunker.py` — `chunk_document` |
| Chunk overlap | 15% | `chunker.py` — `chunk_document` |
| Retrieval depth | top-5 chunks by cosine similarity | `answer.py` — `answer_question(k=5)` |
| Refusal threshold | top score < 0.30 cosine → refuse without an LLM call | `answer.py` — `REFUSAL_THRESHOLD` |
| Embedding batch | ≤ 64 texts/request | `providers/gemini.py` — `EMBED_BATCH` |
| Provider retry | 429 and 5xx → 2s/4s/8s backoff; other 4xx fail fast | `providers/_http.py` — `post_with_retry` |

## Models

| Model | Role | Provider | Price (as coded) | Env key |
|---|---|---|---|---|
| `gemini-3.1-flash-lite` | chat — CLI default | Gemini REST | $0.00 in / $0.00 out | `GEMINI_API_KEY` |
| `gemini-3.5-flash` | chat — quality-comparison row | Gemini REST | $0.00 in / $0.00 out | `GEMINI_API_KEY` |
| `gemini-embedding-001` | embeddings | Gemini REST | _(embed registry carries no price field)_ | `GEMINI_API_KEY` |

Both chat models are coded `$0.00` in `providers/registry.py` — Gemini's free tier. `doclens
models` prints all configured chat models and marks the ones you can use right now with `*` (its
env key is set).

## Scope

This is the CLI + eval-harness core — Plan A of the
[design spec](docs/superpowers/specs/2026-07-14-doclens-design.md) — and that's deliberate. The
spec also describes a hosted FastAPI + SSE web app with per-visitor sessions, rate caps, a
frontend and a Docker/Render deploy (Plan B); none of that is here yet, on purpose, so ingestion,
chunking, the index, the provider adapters and the eval harness each got done properly instead of
five things done halfway.

**Sibling project:** [repolens](https://github.com/nagendernss/repolens) asks questions about
GitHub repos the same way doclens asks questions about documents — same provider-adapter pattern
(raw httpx, no SDKs), same eval-first philosophy (deterministic grading, a runner that splices its
own README table), same author. Where doclens is single-pass top-k retrieval over a document you
hand it, repolens is a multi-step tool-calling agent over a codebase — different retrieval shape,
same engineering standards.

## Project layout

```
doclens/
├── ingest.py              PDF bytes / text → Document (pypdf)
├── ingest_url.py          URL/HTML → Document (httpx + selectolax), SSRF guard
├── chunker.py             Document → list[Chunk], sliding window + sentence snap
├── index.py               VectorIndex — hand-built cosine search (numpy)
├── answer.py              question → retrieve → grounded prompt → AnswerResult
├── types.py               shared dataclasses + fingerprint()
├── cli.py                 `doclens ask` / `doclens models`
└── providers/
    ├── _http.py           shared retry/backoff POST
    ├── registry.py        model table, env-key lookup
    └── gemini.py          chat + batched embeddings (raw REST)

evals/
├── corpus/                3 original authored documents (19 chunks total)
├── gold.yaml              30 cases: 24 answerable + 6 unanswerable
├── metrics.py             recall@5, MRR, faithful, load_gold
├── run.py                 resumable eval runner
└── report.py              results.json → markdown → README splice

tests/                     56 tests, offline (httpx.MockTransport, no live network)
```

## Development

```bash
pip install -e .[dev]
ruff check .
python -m pytest -q                 # 56 tests, no network calls

python -m evals.run --models gemini-3.1-flash-lite --out results.json
python -m evals.report results.json --readme README.md    # fills in the Evals table
```

`.github/workflows/ci.yml` runs the same lint + test steps on every push and pull request.

## License

MIT © 2026 [Nagender Swaroop Srivastava](https://github.com/nagendernss) — see [LICENSE](LICENSE).
