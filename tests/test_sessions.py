"""Tests for in-memory session store with TTL and capacity caps."""

from doclens.index import VectorIndex
from doclens.sessions import SessionDoc, SessionError, SessionStore
from doclens.types import Chunk


def make_chunk(chunk_id: str, doc_id: str, page: int) -> Chunk:
    """Helper to create a test chunk."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        page=page,
        seq=0,
        text=f"Test chunk {chunk_id}",
    )


def make_session_doc(doc_id: str, title: str, pages: int, num_chunks: int) -> SessionDoc:
    """Helper to create a test SessionDoc."""
    chunks = [make_chunk(f"c{i}", doc_id, i % pages) for i in range(num_chunks)]
    return SessionDoc(
        doc_id=doc_id,
        title=title,
        pages=pages,
        chunks=chunks,
        index=VectorIndex(),
        created=0.0,
    )


def test_add_get_roundtrip():
    """Test adding and retrieving a document from a session."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500, now=lambda: clock["t"])

    # Create a new session
    sid = store.new_sid()
    assert sid is not None
    assert len(sid) == 32

    # Add a document
    doc = make_session_doc("doc1", "Test Doc", 5, 100)
    store.add(sid, doc)

    # Retrieve it
    retrieved = store.get(sid, "doc1")
    assert retrieved is not None
    assert retrieved.doc_id == "doc1"
    assert retrieved.title == "Test Doc"
    assert retrieved.pages == 5
    assert len(retrieved.chunks) == 100


def test_unknown_sid_returns_none():
    """Test that unknown session ID returns None."""
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500)
    result = store.get("unknown_sid", "doc1")
    assert result is None


def test_unknown_doc_in_known_sid_returns_none():
    """Test that unknown doc_id in known session returns None."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500, now=lambda: clock["t"])

    sid = store.new_sid()
    doc = make_session_doc("doc1", "Test Doc", 5, 100)
    store.add(sid, doc)

    # Try to get a different doc_id
    result = store.get(sid, "doc2")
    assert result is None


def test_new_sid_uniqueness():
    """Test that new_sid() generates unique session IDs."""
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500)
    sids = {store.new_sid() for _ in range(100)}
    assert len(sids) == 100  # All unique


def test_new_sid_hex_format_and_length():
    """Test that new_sid() returns 32-character hex strings."""
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500)
    for _ in range(10):
        sid = store.new_sid()
        assert len(sid) == 32
        assert all(c in "0123456789abcdef" for c in sid)


def test_max_docs_evicts_oldest():
    """Test that adding beyond max_docs evicts the oldest document."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500, now=lambda: clock["t"])

    sid = store.new_sid()

    # Add 3 documents
    clock["t"] = 100.0
    doc1 = make_session_doc("doc1", "Doc 1", 1, 50)
    store.add(sid, doc1)

    clock["t"] = 200.0
    doc2 = make_session_doc("doc2", "Doc 2", 1, 50)
    store.add(sid, doc2)

    clock["t"] = 300.0
    doc3 = make_session_doc("doc3", "Doc 3", 1, 50)
    store.add(sid, doc3)

    # Verify all 3 exist
    assert store.get(sid, "doc1") is not None
    assert store.get(sid, "doc2") is not None
    assert store.get(sid, "doc3") is not None

    # Add a 4th document (should evict doc1, the oldest)
    clock["t"] = 400.0
    doc4 = make_session_doc("doc4", "Doc 4", 1, 50)
    store.add(sid, doc4)

    # doc1 should be evicted
    assert store.get(sid, "doc1") is None
    assert store.get(sid, "doc2") is not None
    assert store.get(sid, "doc3") is not None
    assert store.get(sid, "doc4") is not None


def test_chunk_budget_raises_error():
    """Test that exceeding chunk budget raises SessionError."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500, now=lambda: clock["t"])

    sid = store.new_sid()

    # Add a document with 1000 chunks
    doc1 = make_session_doc("doc1", "Doc 1", 1, 1000)
    store.add(sid, doc1)

    # Try to add another document that would exceed the budget
    doc2 = make_session_doc("doc2", "Doc 2", 1, 600)
    try:
        store.add(sid, doc2)
        assert False, "Expected SessionError"
    except SessionError as e:
        assert "session chunk budget exceeded" in str(e)

    # Verify doc2 was not added
    assert store.get(sid, "doc2") is None
    assert store.get(sid, "doc1") is not None


def test_sweep_removes_idle_sessions():
    """Test that sweep() removes sessions idle longer than TTL."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500, now=lambda: clock["t"])

    sid = store.new_sid()
    doc = make_session_doc("doc1", "Doc 1", 1, 100)

    # Add document at t=100
    store.add(sid, doc)

    # Fast forward to t=2000 (1900 seconds later, past TTL of 1800)
    clock["t"] = 2000.0
    store.sweep()

    # Session should be removed
    result = store.get(sid, "doc1")
    assert result is None


def test_get_updates_last_access_prevents_sweep():
    """Test that get() touches last-access time and prevents sweep."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500, now=lambda: clock["t"])

    sid = store.new_sid()
    doc = make_session_doc("doc1", "Doc 1", 1, 100)

    # Add document at t=100
    store.add(sid, doc)

    # Access the document at t=1000 (900 seconds later)
    clock["t"] = 1000.0
    retrieved = store.get(sid, "doc1")
    assert retrieved is not None

    # At t=2800 (1800 seconds after last access), it should still exist
    clock["t"] = 2800.0
    result = store.get(sid, "doc1")
    assert result is not None

    # But at t=4601 (1801 seconds after the last access at t=2800), it should be swept
    clock["t"] = 4601.0
    result = store.get(sid, "doc1")
    assert result is None


def test_sweep_called_lazily_on_add():
    """Test that sweep() is called lazily at the start of add()."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500, now=lambda: clock["t"])

    # Create two sessions
    sid1 = store.new_sid()
    sid2 = store.new_sid()

    # Add documents
    clock["t"] = 100.0
    doc1 = make_session_doc("doc1", "Doc 1", 1, 100)
    store.add(sid1, doc1)

    # Add sid2 much later
    clock["t"] = 1500.0
    doc2 = make_session_doc("doc2", "Doc 2", 1, 100)
    store.add(sid2, doc2)

    # Fast forward so sid1 is idle (2000 seconds after its last access at t=100)
    # but sid2 is still fresh (600 seconds after its last access at t=1500)
    clock["t"] = 2100.0  # 2000 seconds after sid1, 600 after sid2

    # Add to sid2 (should trigger sweep, removing sid1 but not sid2)
    doc3 = make_session_doc("doc3", "Doc 3", 1, 100)
    store.add(sid2, doc3)

    # sid1 should be swept
    result = store.get(sid1, "doc1")
    assert result is None

    # sid2 should still exist
    result = store.get(sid2, "doc2")
    assert result is not None


def test_sweep_called_lazily_on_get():
    """Test that sweep() is called lazily at the start of get()."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500, now=lambda: clock["t"])

    sid = store.new_sid()
    doc = make_session_doc("doc1", "Doc 1", 1, 100)

    # Add document at t=100
    store.add(sid, doc)

    # Fast forward past TTL
    clock["t"] = 2100.0

    # get() should trigger sweep
    result = store.get(sid, "doc1")
    assert result is None


def test_multiple_sessions_isolated():
    """Test that documents in different sessions are isolated."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500, now=lambda: clock["t"])

    sid1 = store.new_sid()
    sid2 = store.new_sid()

    # Add same doc_id to both sessions
    clock["t"] = 100.0
    doc_sid1 = make_session_doc("doc1", "Doc 1 Session 1", 1, 100)
    store.add(sid1, doc_sid1)

    clock["t"] = 200.0
    doc_sid2 = make_session_doc("doc1", "Doc 1 Session 2", 1, 100)
    store.add(sid2, doc_sid2)

    # Retrieve from each session
    retrieved_sid1 = store.get(sid1, "doc1")
    retrieved_sid2 = store.get(sid2, "doc1")

    assert retrieved_sid1.title == "Doc 1 Session 1"
    assert retrieved_sid2.title == "Doc 1 Session 2"


def test_evicted_doc_frees_chunk_budget():
    """Test that evicting a document frees up doc slots for new documents."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=2, max_chunks=1500, now=lambda: clock["t"])

    sid = store.new_sid()

    # Add doc with 700 chunks
    clock["t"] = 100.0
    doc1 = make_session_doc("doc1", "Doc 1", 1, 700)
    store.add(sid, doc1)

    # Add doc with 600 chunks (total 1300, under budget)
    clock["t"] = 200.0
    doc2 = make_session_doc("doc2", "Doc 2", 1, 600)
    store.add(sid, doc2)

    # Now we have 2 docs (at max_docs=2), so adding a 3rd should evict doc1
    clock["t"] = 300.0
    doc3 = make_session_doc("doc3", "Doc 3", 1, 400)
    store.add(sid, doc3)

    # doc1 should be evicted (oldest), doc2 and doc3 should remain
    assert store.get(sid, "doc1") is None
    assert store.get(sid, "doc2") is not None
    assert store.get(sid, "doc3") is not None


def test_chunk_budget_with_eviction():
    """Test that chunk budget respects eviction when hitting max_docs."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=2, max_chunks=1000, now=lambda: clock["t"])

    sid = store.new_sid()

    # Add doc1 with 600 chunks
    clock["t"] = 100.0
    doc1 = make_session_doc("doc1", "Doc 1", 1, 600)
    store.add(sid, doc1)

    # Add doc2 with 400 chunks (total 1000, at budget)
    clock["t"] = 200.0
    doc2 = make_session_doc("doc2", "Doc 2", 1, 400)
    store.add(sid, doc2)

    # Try to add doc3 with 200 chunks
    # This exceeds max_docs=2, so doc1 (oldest) should be evicted first
    # After eviction, we have 400 chunks (doc2) + 200 (doc3) = 600, well under budget
    clock["t"] = 300.0
    doc3 = make_session_doc("doc3", "Doc 3", 1, 200)
    store.add(sid, doc3)

    assert store.get(sid, "doc1") is None
    assert store.get(sid, "doc2") is not None
    assert store.get(sid, "doc3") is not None


def test_thread_safe_operations():
    """Test basic thread-safety with multiple threads accessing the store."""
    import threading

    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=10, max_chunks=5000, now=lambda: clock["t"])

    sid = store.new_sid()
    errors = []

    def add_docs():
        try:
            for i in range(5):
                doc = make_session_doc(f"doc_{threading.current_thread().name}_{i}", f"Doc {i}", 1, 100)
                store.add(sid, doc)
        except Exception as e:
            errors.append(e)

    # Create multiple threads adding documents
    threads = [threading.Thread(target=add_docs, name=f"thread_{i}") for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent add: {errors}"

    # Verify some documents exist
    assert store.get(sid, "doc_thread_0_0") is not None or store.get(sid, "doc_thread_1_0") is not None


def test_created_timestamp_preserved():
    """Test that created timestamp is preserved in retrieved SessionDoc."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500, now=lambda: clock["t"])

    sid = store.new_sid()
    doc = SessionDoc(
        doc_id="doc1",
        title="Test Doc",
        pages=5,
        chunks=[make_chunk("c0", "doc1", 0)],
        index=VectorIndex(),
        created=42.5,
    )
    store.add(sid, doc)

    retrieved = store.get(sid, "doc1")
    assert retrieved.created == 42.5


def test_empty_store_has_no_sessions():
    """Test that empty store returns None for any get."""
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=1500)
    assert store.get("any_sid", "any_doc") is None


def test_session_error_message():
    """Test that SessionError is raised with correct message."""
    clock = {"t": 100.0}
    store = SessionStore(ttl_s=1800, max_docs=3, max_chunks=10, now=lambda: clock["t"])

    sid = store.new_sid()

    # Add document that uses all budget
    doc1 = make_session_doc("doc1", "Doc 1", 1, 10)
    store.add(sid, doc1)

    # Try to add another document (should fail)
    doc2 = make_session_doc("doc2", "Doc 2", 1, 1)
    try:
        store.add(sid, doc2)
        assert False, "Expected SessionError"
    except SessionError as e:
        assert str(e) == "session chunk budget exceeded"
