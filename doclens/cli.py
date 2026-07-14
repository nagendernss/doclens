"""doclens CLI: one-shot ingest + ask from the terminal."""
from __future__ import annotations

import argparse
import sys

from .answer import answer_question
from .chunker import chunk_document
from .index import VectorIndex
from .ingest import IngestError, ingest_file
from .ingest_url import ingest_url
from .providers.registry import (CHAT_MODELS, MissingKeyError, UnknownModelError,
                                 available_chat_models, get_chat, get_embedder)

DEFAULT_MODEL = "gemini-3.1-flash-lite"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(prog="doclens")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("source")
    ask.add_argument("question")
    ask.add_argument("--model", default=DEFAULT_MODEL)
    ask.add_argument("-k", type=int, default=5)
    sub.add_parser("models")
    args = parser.parse_args(argv)

    if args.cmd == "models":
        avail = set(available_chat_models())
        for name in CHAT_MODELS:
            print(f"{name} *" if name in avail else name)
        return 0

    try:
        chat, chat_model = get_chat(args.model)
        embedder, embed_model = get_embedder()
    except (MissingKeyError, UnknownModelError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        source = args.source
        doc = ingest_url(source) if source.startswith(("http://", "https://")) \
            else ingest_file(source)
        chunks = chunk_document(doc)
        vectors = embedder.embed([c.text for c in chunks], embed_model)
        index = VectorIndex()
        index.add(chunks, vectors)
        print(f"doclens · {doc.title} · {len(doc.pages)} pages · {len(chunks)} chunks")
        res = answer_question(chat, chat_model, embedder, embed_model, index,
                              args.question, k=args.k)
        pages = ", ".join(f"p.{r.chunk.page}" for r in res.retrieved)
        print(f"retrieved: {pages}")
        print()
        print(res.answer)
        print(f"\nmodel={args.model} tokens={res.usage.input_tokens}+{res.usage.output_tokens}")
        return 0
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # provider/network
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
