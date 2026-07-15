"""Hand-built BM25 lexical index for doclens retrieval."""
from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doclens.types import Chunk


STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "can", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "just", "me", "my", "no", "not", "of", "on",
    "or", "our", "out", "own", "she", "so", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to",
    "too", "up", "was", "we", "what", "when", "which", "who", "why",
    "will", "with", "you", "your"
})


def _tokenize(text: str) -> list[str]:
    """Tokenize text: lowercase, extract alphanumeric, drop stopwords and short tokens.

    Args:
        text: Raw text to tokenize

    Returns:
        List of tokens (lowercase, length >= 2, excluding stopwords)
    """
    text_lower = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text_lower)
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]


class BM25Index:
    """BM25 lexical ranking index.

    Supports incremental document addition and Okapi BM25 ranking with
    configurable k1 and b parameters.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 index.

        Args:
            k1: Term saturation parameter (default 1.5)
            b: Length normalization parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(chunk_idx, tf)]
        self._df: dict[str, int] = {}  # term -> document frequency
        self._len: list[int] = []  # chunk_idx -> token count
        self._n: int = 0  # total chunks
        self._sum_len: int = 0  # sum of all chunk lengths

    def __len__(self) -> int:
        """Return number of chunks in the index."""
        return self._n

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index.

        Args:
            chunks: List of Chunk objects to index
        """
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            chunk_idx = self._n
            self._n += 1
            self._len.append(len(tokens))
            self._sum_len += len(tokens)

            # Count term frequencies in this chunk
            tf_dict: dict[str, int] = {}
            for token in tokens:
                tf_dict[token] = tf_dict.get(token, 0) + 1

            # Update postings and document frequency
            for term, tf in tf_dict.items():
                if term not in self._postings:
                    self._postings[term] = []
                    self._df[term] = 0
                self._postings[term].append((chunk_idx, tf))
                self._df[term] += 1

    def rank(self, query: str) -> list[tuple[int, float]]:
        """Rank chunks by BM25 score for the query.

        Args:
            query: Query text

        Returns:
            List of (chunk_idx, score) tuples sorted by score descending,
            then by index ascending. Returns empty list if index is empty
            or query yields no matches.
        """
        if self._n == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        avglen = self._sum_len / self._n  # _n > 0 guaranteed by the guard above
        scores: dict[int, float] = {}

        for token in query_tokens:
            if token not in self._postings:
                continue

            df = self._df[token]
            idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

            for chunk_idx, tf in self._postings[token]:
                chunk_len = self._len[chunk_idx]
                norm = 1.0 - self.b + self.b * (chunk_len / avglen if avglen > 0 else 0)
                score_component = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
                scores[chunk_idx] = scores.get(chunk_idx, 0.0) + score_component

        # Filter out zero scores and sort by score desc, index asc
        ranked = [(idx, score) for idx, score in scores.items() if score > 0]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        return ranked
