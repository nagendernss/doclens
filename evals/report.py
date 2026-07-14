"""Report: summarize eval results and splice into README."""
from __future__ import annotations

import argparse
import json
from statistics import median


def summarize(records: list[dict]) -> list[dict]:
    """Summarize records per model: recall_at_5, mrr, faithfulness, refusal_acc, p50_latency.

    Args:
        records: List of eval records (from run_eval output).

    Returns:
        List of summary dicts per model, sorted by model name.

    """
    by_model = {}
    for r in records:
        model = r["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    summaries = []
    for model in sorted(by_model.keys()):
        recs = by_model[model]

        # recall_at_5: mean over answerable cases (where recall5 is not None)
        recall5_vals = [r["recall5"] for r in recs if r["recall5"] is not None]
        recall_at_5 = sum(recall5_vals) / len(recall5_vals) if recall5_vals else None

        # mrr: mean over answerable cases
        mrr_vals = [r["mrr"] for r in recs if r["mrr"] is not None]
        mrr_val = sum(mrr_vals) / len(mrr_vals) if mrr_vals else None

        # faithfulness: fraction of answerable cases with faithful=True
        faithful_vals = [r["faithful"] for r in recs if r["faithful"] is not None]
        faithfulness = (sum(faithful_vals) / len(faithful_vals)
                        if faithful_vals else None)

        # refusal_acc: fraction of unanswerable cases with refused_correctly=True
        refused_vals = [r["refused_correctly"] for r in recs
                       if r["refused_correctly"] is not None]
        refusal_acc = sum(refused_vals) / len(refused_vals) if refused_vals else None

        # p50_latency: median latency_s
        latencies = sorted(r["latency_s"] for r in recs if r["latency_s"] is not None)
        p50_latency = median(latencies) if latencies else None

        summaries.append({
            "model": model,
            "recall_at_5": recall_at_5,
            "mrr": mrr_val,
            "faithfulness": faithfulness,
            "refusal_acc": refusal_acc,
            "p50_latency": p50_latency,
        })

    return summaries


def to_markdown(summaries: list[dict]) -> str:
    """Render summaries as markdown table.

    Header: | Model | Recall@5 | MRR | Faithful | Refusal acc | p50 s |

    """
    lines = [
        "| Model | Recall@5 | MRR | Faithful | Refusal acc | p50 s |",
        "|-------|----------|-----|----------|-------------|-------|",
    ]

    for s in summaries:
        model = s["model"]
        recall = f"{s['recall_at_5']:.2f}" if s["recall_at_5"] is not None else "—"
        mrr_str = f"{s['mrr']:.2f}" if s["mrr"] is not None else "—"
        faithful = f"{s['faithfulness']:.1%}" if s["faithfulness"] is not None else "—"
        refusal = f"{s['refusal_acc']:.1%}" if s["refusal_acc"] is not None else "—"
        p50 = f"{s['p50_latency']:.2f}" if s["p50_latency"] is not None else "—"
        lines.append(f"| {model} | {recall} | {mrr_str} | {faithful} | {refusal} | {p50} |")

    return "\n".join(lines)


def splice_readme(content: str, table: str) -> str:
    """Splice table between <!-- evals:start --> and <!-- evals:end --> markers.

    Args:
        content: README content.
        table: Markdown table to splice in.

    Returns:
        Updated content with table spliced between markers.

    Raises:
        ValueError: if markers are missing.

    """
    start_marker = "<!-- evals:start -->"
    end_marker = "<!-- evals:end -->"

    if start_marker not in content or end_marker not in content:
        raise ValueError(f"missing markers (need both {start_marker!r} and {end_marker!r})")

    start_idx = content.index(start_marker)
    end_idx = content.index(end_marker)
    if start_idx >= end_idx:
        raise ValueError("markers out of order")

    before = content[:start_idx + len(start_marker)]
    after = content[end_idx:]
    return f"{before}\n{table}\n{after}"


def main():
    """CLI: python -m evals.report results.json [--readme README.md]"""
    parser = argparse.ArgumentParser(description="Summarize eval results.")
    parser.add_argument("results", type=str, help="Results JSON path.")
    parser.add_argument("--readme", type=str, default="README.md",
                        help="README path to splice into (default: README.md).")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as fh:
        results = json.load(fh)

    records = results.get("records", [])
    summaries = summarize(records)
    table = to_markdown(summaries)
    print(table)

    # Optionally splice into README
    try:
        with open(args.readme, encoding="utf-8") as fh:
            readme = fh.read()
        updated = splice_readme(readme, table)
        with open(args.readme, "w", encoding="utf-8") as fh:
            fh.write(updated)
        print(f"\nSpliced into {args.readme}")
    except FileNotFoundError:
        print(f"\n{args.readme} not found; skipping splice")
    except ValueError as e:
        print(f"\n{args.readme}: {e}; skipping splice")


if __name__ == "__main__":
    main()
