# When LLM Reranking *Hurt* My RAG (and the Eval That Caught It)

**TL;DR** — I added a two-stage retriever to my RAG app: hybrid search (dense + BM25) followed by an LLM listwise reranker. The reranker was supposed to be the win. Instead the eval showed it dropped mean reciprocal rank from **0.86 to 0.68** — it was actively demoting chunks that plain cosine similarity had already ranked correctly. I shipped it as an opt-in mode, defaulted to plain hybrid, and wrote down why. This is that write-up.

---

## The setup

[doclens](https://doclens-05fb.onrender.com) is a small RAG app I built from scratch — no LangChain, no vector-DB SaaS. You upload a PDF or paste a URL, it chunks and embeds the text (Gemini embeddings), stores vectors in a hand-written numpy cosine index, and answers questions with page-level citations.

The baseline retriever was pure dense cosine similarity. I wanted to make it better, so I added the two upgrades every "advanced RAG" guide recommends:

1. **Hybrid search** — fuse the dense ranking with a hand-built BM25 lexical ranking using Reciprocal Rank Fusion, so exact-term matches (error codes, IDs, rare tokens) that embeddings blur still surface.
2. **LLM listwise reranking** — take the top ~20 fused candidates and ask the LLM to reorder them by relevance before the answer is generated.

The reranker is the piece everyone treats as a guaranteed upgrade. I assumed the same.

## The eval

I already had an eval harness with deterministic grading — no LLM judge. 36 gold cases (29 answerable + 7 unanswerable) across 4 documents, graded on:

- **Recall@5** — is a relevant chunk in the top 5?
- **MRR** — reciprocal rank of the *first* relevant chunk (does the best chunk rank first, not just land in the top 5?)
- **Faithfulness** — does every claim's citation point at a page that was actually retrieved?
- **Refusal accuracy** — does it correctly say "not in the document" when it should?

I added a `mode` axis so I could run `dense`, `hybrid`, and `hybrid_rerank` side by side on the exact same cases. All on `gemini-3.1-flash-lite`.

## The result I did not expect

| Mode | Recall@5 | MRR | Faithful | Refusal | p50 latency |
|------|----------|-----|----------|---------|-------------|
| dense | 1.00 | **0.865** | 100% | 100% | 1.33s |
| hybrid | 1.00 | 0.833 | 100% | 100% | 1.34s |
| hybrid_rerank | 1.00 | **0.683** | 100% | 100% | 2.07s |

Recall@5 is saturated at 1.00 — with only a handful of chunks per doc and `k=5`, almost everything relevant comes back regardless, so recall isn't discriminating here. MRR is the signal. And MRR goes the *wrong way*: every stage I added made it worse, and the reranker — the supposed headline feature — made it worse by a lot, at 1.6× the latency.

## Why — the part that's actually interesting

Aggregates hide things, so I split MRR by query type:

| Query type | dense | hybrid | hybrid_rerank |
|------------|-------|--------|---------------|
| semantic (24 cases) | 0.858 | 0.819 | **0.638** |
| exact-term (5 probe cases) | 0.900 | 0.900 | 0.900 |

Now it's legible. On the **exact-term** probes — questions like "what does diagnostic code KR-4021 mean?" against a doc full of near-identical error-code entries — reranking is a wash. It pulls one needle up (a case where dense buried the exact match at rank 2 → rank 1) and knocks another one down. Net zero.

On the **semantic** cases it's a small disaster: **0.86 → 0.64**. Ten cases regressed, several from a perfect 1.0 down to 0.2–0.5 — the reranker took the correct chunk that cosine had ranked first and shoved it to rank 3, 4, 5.

The mechanism is mundane once you see it: on a semantic query where dense retrieval is *already good*, a small, cheap model asked to reorder 20 similar-looking passages doesn't have a strong enough signal to beat cosine — so it adds noise. Reranking only helps when the first-stage ranking is weak or when there's a lexical/exact-match signal the embedding genuinely missed. On a semantic-heavy corpus with a strong dense baseline, there's nothing for it to fix and plenty for it to break.

## What I shipped

I did not delete the reranker, and I did not pretend the numbers were good.

- **Default mode: `hybrid`.** It costs ~0.03 MRR versus dense but keeps the BM25 lexical channel, which matters more as a corpus grows and on exact-term queries this small eval set under-represents.
- **`hybrid_rerank` stays available per request** — for corpora that are exact-term-heavy, or paired with a stronger judge model.
- **The README documents the whole tradeoff**, per-type table included, so anyone reading it sees when reranking would actually pay off.
- The score bar in the UI shows the calibrated dense cosine, and refusal is decided on the dense score *before* reranking — so the reranker can never change whether the app refuses.

## Takeaways

- **Rerankers are not a free upgrade.** They help when first-stage retrieval is weak or misses a lexical signal. On a strong dense baseline they can be pure noise — and a small reranker model makes that worse.
- **Measure per-slice, not just in aggregate.** The headline "MRR dropped" was true but shallow; the per-type split ("helps exact-term, hurts semantic") is the actual finding and the actual product decision.
- **Recall@k saturates on small corpora.** If your eval set is small, lean on MRR and faithfulness, and say so out loud.
- **Ship the honest result.** A measured negative with a data-driven default is more trustworthy — and more useful to the next engineer — than a benchmark cherry-picked to look like a win.

## Reproduce it

```bash
python -m evals.run --models gemini-3.1-flash-lite --modes dense,hybrid,hybrid_rerank --out results.json
python -m evals.report results.json --readme README.md
```

Code: [github.com/nagendernss/doclens](https://github.com/nagendernss/doclens) · Live: [doclens-05fb.onrender.com](https://doclens-05fb.onrender.com)
