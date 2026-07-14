"""In-memory per-visitor session store with TTL and capacity caps."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass

from .index import VectorIndex
from .types import Chunk


class SessionError(Exception):
    """Raised when session operation violates constraints."""

    pass


@dataclass
class SessionDoc:
    """A document stored in a session."""

    doc_id: str
    title: str
    pages: int
    chunks: list[Chunk]
    index: VectorIndex
    created: float


class SessionStore:
    """Thread-safe in-memory session store with TTL and capacity caps."""

    def __init__(self, ttl_s: float = 1800, max_docs: int = 3, max_chunks: int = 1500, now=None):
        """Initialize the session store.

        Args:
            ttl_s: Time-to-live for idle sessions in seconds. Default 1800 (30 min).
            max_docs: Maximum documents per session. Default 3.
            max_chunks: Maximum total chunks across all docs in a session. Default 1500.
            now: Callable returning current time (seconds). Defaults to time.time().
        """
        self.ttl_s = ttl_s
        self.max_docs = max_docs
        self.max_chunks = max_chunks
        self.now = now or time.time

        self.lock = threading.Lock()
        # Structure: sessions[sid] = {"last_access": float, "docs": {doc_id: SessionDoc}}
        self.sessions: dict[str, dict] = {}

    def new_sid(self) -> str:
        """Generate a new 32-character hex session ID.

        Returns:
            A random 32-character hex string.
        """
        return secrets.token_hex(16)

    def _sweep_unlocked(self) -> None:
        """Remove sessions idle longer than TTL (must be called with lock held)."""
        current_time = self.now()
        sids_to_remove = [
            sid
            for sid, session_data in self.sessions.items()
            if current_time - session_data["last_access"] > self.ttl_s
        ]
        for sid in sids_to_remove:
            del self.sessions[sid]

    def sweep(self) -> None:
        """Remove sessions idle longer than TTL."""
        with self.lock:
            self._sweep_unlocked()

    def add(self, sid: str, doc: SessionDoc) -> None:
        """Add a document to a session.

        Enforces max_docs cap by evicting oldest doc if needed.
        Enforces max_chunks cap by raising SessionError.

        Args:
            sid: Session ID.
            doc: SessionDoc to add.

        Raises:
            SessionError: If total chunks would exceed max_chunks after eviction.
        """
        with self.lock:
            # Lazy sweep
            self._sweep_unlocked()

            # Ensure session exists
            if sid not in self.sessions:
                self.sessions[sid] = {"last_access": self.now(), "docs": {}}

            session_data = self.sessions[sid]
            docs = session_data["docs"]

            # Check if eviction needed due to max_docs
            if len(docs) >= self.max_docs:
                # Find and evict oldest doc
                oldest_sid = min(docs.keys(), key=lambda d: docs[d].created)
                del docs[oldest_sid]

            # Calculate current chunk count
            current_chunks = sum(len(d.chunks) for d in docs.values())
            new_chunks = len(doc.chunks)

            # Check if adding would exceed max_chunks
            if current_chunks + new_chunks > self.max_chunks:
                raise SessionError("session chunk budget exceeded")

            # Add document
            docs[doc.doc_id] = doc
            session_data["last_access"] = self.now()

    def get(self, sid: str, doc_id: str) -> SessionDoc | None:
        """Retrieve a document from a session.

        Touches last-access time for the session.

        Args:
            sid: Session ID.
            doc_id: Document ID within the session.

        Returns:
            SessionDoc if found and session not swept, None otherwise.
        """
        with self.lock:
            # Lazy sweep
            self._sweep_unlocked()

            # Check if session exists
            if sid not in self.sessions:
                return None

            session_data = self.sessions[sid]
            docs = session_data["docs"]

            # Check if doc exists
            if doc_id not in docs:
                return None

            # Touch last-access time
            session_data["last_access"] = self.now()

            return docs[doc_id]
