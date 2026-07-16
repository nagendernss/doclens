"""Reciprocal Rank Fusion for retrieval ranking aggregation."""
from __future__ import annotations


def rrf(rankings: list[list[int]], k_const: int = 60) -> list[tuple[int, float]]:
    """
    Fuse multiple rankings using reciprocal rank fusion.

    Each ranking is a best-first list of chunk indices. The score for each index
    is computed as the sum of 1 / (k_const + rank) across all rankings where the
    index appears (1-based rank within each list). Indices not present in a
    ranking contribute 0 from that list.

    Args:
        rankings: List of best-first rank lists (each element is a list of chunk indices).
        k_const: Constant for RRF formula (default 60).

    Returns:
        Deduped (idx, score) tuples sorted by score descending, then idx ascending
        as a tiebreaker. Empty input returns [].
    """
    if not rankings or all(not r for r in rankings):
        return []

    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            score_contribution = 1.0 / (k_const + rank)
            scores[idx] = scores.get(idx, 0.0) + score_contribution

    # Sort by score descending, then idx ascending
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
