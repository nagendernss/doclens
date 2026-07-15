"""In-memory per-visitor session store with TTL and capacity caps."""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .index import VectorIndex
from .types import Chunk

if TYPE_CHECKING:
    from .hybrid import HybridIndex
    from .trace import Trace

MAX_TRACES = 200


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
    index: HybridIndex
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
        self._traces: OrderedDict[str, Trace] = OrderedDict()

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
        All computation is done before mutation to ensure atomicity on error.

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

            # Step 1: Determine if doc.doc_id already exists (update vs net-new)
            is_update = doc.doc_id in docs

            # Step 2: Determine the victim to evict ONLY if net-new AND len(docs) >= max_docs
            victim_id = None
            if not is_update and len(docs) >= self.max_docs:
                victim_id = min(docs.keys(), key=lambda d: docs[d].created)

            # Step 3: Compute the resulting chunk total BEFORE mutating
            # Start with sum of existing chunks
            current_chunks = sum(len(d.chunks) for d in docs.values())

            # Subtract victim's chunks if evicting
            if victim_id is not None:
                current_chunks -= len(docs[victim_id].chunks)

            # Subtract old doc's chunks if updating same doc_id
            if is_update:
                current_chunks -= len(docs[doc.doc_id].chunks)

            # Add new doc chunks
            new_total = current_chunks + len(doc.chunks)

            # Step 4: If resulting total > max_chunks, raise SessionError with NO mutation
            if new_total > self.max_chunks:
                raise SessionError("session chunk budget exceeded")

            # Step 5: Only then mutate (delete victim if any, assign docs[doc.doc_id] = doc)
            if victim_id is not None:
                del docs[victim_id]

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

    def add_trace(self, trace: Trace) -> None:
        """Store a trace in the bounded ring.

        Evicts the oldest trace when exceeding MAX_TRACES.

        Args:
            trace: Trace object to store.
        """
        with self.lock:
            self._traces[trace.trace_id] = trace
            if len(self._traces) > MAX_TRACES:
                self._traces.popitem(last=False)

    def get_trace(self, trace_id: str) -> Trace | None:
        """Retrieve a stored trace by ID.

        Args:
            trace_id: Trace ID to look up.

        Returns:
            Trace if found, None otherwise.
        """
        with self.lock:
            return self._traces.get(trace_id)
