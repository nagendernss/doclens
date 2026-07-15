"""Eval runner: compute retrieval and faithfulness metrics over gold cases."""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

from doclens.answer import answer_question
from doclens.chunker import chunk_document
from doclens.hybrid import HybridIndex
from doclens.ingest import ingest_file
from doclens.providers.registry import get_chat, get_embedder
from doclens.types import fingerprint
from evals.metrics import faithful, load_gold, mrr, recall_at_k


def run_eval(
    models: list[str],
    gold: list[dict],
    out_path: str,
    *,
    embedder_factory=get_embedder,
    chat_factory=get_chat,
    sleep_s: float = 2.0,
) -> dict:
    """Run eval over gold cases, resume by (model, case_id), atomic writes.

    Args:
        models: List of chat model names.
        gold: List of gold cases from load_gold.
        out_path: Path to results.json (create if missing).
        embedder_factory: Callable that returns (embedder, model_name).
        chat_factory: Callable that returns (chat, model_name).
        sleep_s: Sleep between requests (for rate limiting).

    Returns:
        {"records": [record, ...], "metadata": {...}}.

    """
    # Load or initialize results
    try:
        with open(out_path, encoding="utf-8") as fh:
            results = json.load(fh)
    except FileNotFoundError:
        results = {"records": [], "metadata": {}}
    except json.JSONDecodeError:
        warnings.warn(f"{out_path}: corrupt or malformed; starting fresh")
        results = {"records": [], "metadata": {}}

    # Validate that results is a dict with records key
    if not isinstance(results, dict) or "records" not in results:
        warnings.warn(f"{out_path}: corrupt or malformed; starting fresh")
        results = {"records": [], "metadata": {}}

    # Build resume set: {(model, case_id) for already-run}
    resume_set = {(r["model"], r["case_id"]) for r in results["records"]}

    # Compute pending (model, case_id) pairs
    pending_pairs = set()
    for model in models:
        for case in gold:
            if (model, case["id"]) not in resume_set:
                pending_pairs.add((model, case["id"]))

    # If no pending cases, return early (no need to build corpus cache or embed)
    if not pending_pairs:
        return results

    # Build corpus cache: {doc_filename: (chunks, index)}
    # Only for docs referenced by pending cases
    pending_case_ids = {pair[1] for pair in pending_pairs}
    docs_needed = {case["doc"] for case in gold if case["id"] in pending_case_ids}

    corpus_cache = {}
    embedder, embed_model = embedder_factory()

    for doc_filename in docs_needed:
        doc_path = f"evals/corpus/{doc_filename}"
        doc = ingest_file(doc_path)
        chunks = chunk_document(doc)
        vecs = embedder.embed([c.text for c in chunks], embed_model)
        index = HybridIndex()
        index.add(chunks, vecs)
        corpus_cache[doc_filename] = (chunks, index)

    # Run evals per (model, case)
    for model in models:
        chat, chat_model = chat_factory(model)
        for case in gold:
            if (model, case["id"]) in resume_set:
                continue

            chunks, index = corpus_cache[case["doc"]]
            question = case["question"]
            answerable = case["answerable"]
            expected_facts = case["expected_facts"]
            relevant_fps = case["relevant_fps"]

            # Run answer_question
            t0 = time.time()
            try:
                res = answer_question(chat, chat_model, embedder, embed_model,
                                    index, question, k=5)
                latency_s = time.time() - t0
                error = None
            except Exception as e:
                latency_s = time.time() - t0
                error = str(e)
                res = None

            # Grade the result
            if error:
                record = {
                    "model": model,
                    "case_id": case["id"],
                    "recall5": None,
                    "mrr": None,
                    "faithful": None,
                    "refused_correctly": None,
                    "latency_s": latency_s,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error": error,
                }
            elif answerable:
                # Retrieve: compute fingerprints
                retrieved_fps = [
                    fingerprint(c.chunk.doc_id, c.chunk.page, c.chunk.text)
                    for c in res.retrieved
                ]
                recall5 = recall_at_k(retrieved_fps, relevant_fps)
                mrr_val = mrr(retrieved_fps, relevant_fps)
                retrieved_pages = [c.chunk.page for c in res.retrieved]
                is_faithful = faithful(res.answer, res.citations, retrieved_pages,
                                     expected_facts)

                record = {
                    "model": model,
                    "case_id": case["id"],
                    "recall5": recall5,
                    "mrr": mrr_val,
                    "faithful": is_faithful,
                    "refused_correctly": None,
                    "latency_s": latency_s,
                    "input_tokens": res.usage.input_tokens,
                    "output_tokens": res.usage.output_tokens,
                    "error": None,
                }
            else:
                # Unanswerable: check refusal
                record = {
                    "model": model,
                    "case_id": case["id"],
                    "recall5": None,
                    "mrr": None,
                    "faithful": None,
                    "refused_correctly": res.refused,
                    "latency_s": latency_s,
                    "input_tokens": res.usage.input_tokens,
                    "output_tokens": res.usage.output_tokens,
                    "error": None,
                }

            results["records"].append(record)

            # Atomic write
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)
            os.replace(tmp_path, out_path)

            if sleep_s > 0:
                time.sleep(sleep_s)

    return results


def main():
    """CLI: python -m evals.run --models a,b --out results.json [--gold path] [--sleep 2]"""
    parser = argparse.ArgumentParser(description="Run eval over gold cases.")
    parser.add_argument("--models", type=str, required=True,
                        help="Comma-separated chat models.")
    parser.add_argument("--out", type=str, required=True, help="Output results.json path.")
    parser.add_argument("--gold", type=str, default="evals/gold.yaml",
                        help="Gold cases YAML path (default: evals/gold.yaml).")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Sleep between requests (default: 2.0).")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    gold = load_gold(args.gold)
    results = run_eval(models, gold, args.out, sleep_s=args.sleep)
    print(f"Results: {len(results['records'])} records in {args.out}")


if __name__ == "__main__":
    main()
