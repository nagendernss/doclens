"""HybridIndex: dense ⊕ BM25 retrieval fused by RRF, with full provenance."""
from __future__ import annotations

from .fusion import rrf
from .index import VectorIndex
from .lexical import BM25Index
from .types import Chunk, Retrieved


class HybridIndex:
    """Owns the canonical chunk list plus a dense and a lexical retriever.

    `add` feeds both retrievers the same chunks in the same call, so the
    dense index and the BM25 index share one 0-based chunk-index space;
    `_chunks[idx]` resolves indices from either retriever's rankings.
    """

    def __init__(self) -> None:
        """Initialize empty dense index, lexical index, and chunk list."""
        self.dense = VectorIndex()
        self.lexical = BM25Index()
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        """Return number of indexed chunks."""
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Add chunks to both retrievers, keeping their index spaces aligned.

        Args:
            chunks: list of Chunk objects.
            vectors: list of embedding vectors, one per chunk.

        """
        self.dense.add(chunks, vectors)
        self.lexical.add(chunks)
        self._chunks.extend(chunks)

    def retrieve(self, qvec: list[float], qtext: str, *,
                 mode: str = "hybrid", pool: int = 20) -> list[Retrieved]:
        """Retrieve a candidate pool with full provenance in `components`.

        Args:
            qvec: query embedding for the dense retriever.
            qtext: query text for the lexical retriever.
            mode: "dense", "lexical", or "hybrid" (default).
            pool: maximum number of candidates to return.

        Returns:
            list of Retrieved sorted by the mode's primary score descending.
            Every candidate's `components["dense_score"]` is populated
            regardless of mode, so the refusal gate always has a calibrated
            cosine. `mode == "dense"` returns exactly the same order/scores
            as `VectorIndex.search`. `components["rrf_score"]` is present
            only in hybrid mode; `components["bm25_rank"]` is None for a
            chunk that never scored in BM25.

        """
        dense_ranked = self.dense.rank_all(qvec)
        cos_by_idx = dict(dense_ranked)
        dense_order = [i for i, _ in dense_ranked]
        dense_rank = {i: r for r, i in enumerate(dense_order, 1)}

        if mode == "dense":
            chosen, primary = dense_order[:pool], cos_by_idx
            bm25_rank: dict[int, int] = {}
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
            comp = {
                "dense_score": cos_by_idx.get(idx, 0.0),
                "dense_rank": dense_rank.get(idx),
                "bm25_rank": bm25_rank.get(idx),
            }
            if mode == "hybrid":
                comp["rrf_score"] = primary.get(idx, 0.0)
            out.append(Retrieved(chunk=self._chunks[idx], score=primary.get(idx, 0.0),
                                  components=comp))
        return out
