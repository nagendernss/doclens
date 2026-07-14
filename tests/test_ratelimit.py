from doclens.ratelimit import RateLimiter


def test_per_ip_ingest_cap():
    rl = RateLimiter(per_ip_ingest=2, per_ip_question=10, global_cap=100)
    assert rl.allow("1.1.1.1", "ingest")[0] is True
    assert rl.allow("1.1.1.1", "ingest")[0] is True
    ok, reason = rl.allow("1.1.1.1", "ingest")
    assert ok is False and "daily limit" in reason
    assert rl.allow("1.1.1.1", "question")[0] is True  # separate kind counter


def test_global_cap_combined():
    rl = RateLimiter(per_ip_ingest=10, per_ip_question=10, global_cap=2)
    rl.allow("1.1.1.1", "ingest")
    rl.allow("2.2.2.2", "question")
    ok, reason = rl.allow("3.3.3.3", "ingest")
    assert ok is False and "global" in reason


def test_utc_reset():
    day = {"d": "2026-07-14"}
    rl = RateLimiter(per_ip_ingest=1, per_ip_question=1, global_cap=99, today=lambda: day["d"])
    assert rl.allow("1.1.1.1", "ingest")[0] is True
    assert rl.allow("1.1.1.1", "ingest")[0] is False
    day["d"] = "2026-07-15"
    assert rl.allow("1.1.1.1", "ingest")[0] is True


def test_denied_not_counted():
    rl = RateLimiter(per_ip_ingest=1, per_ip_question=1, global_cap=99)
    rl.allow("1.1.1.1", "ingest")
    rl.allow("1.1.1.1", "ingest")  # denied
    assert rl.remaining("1.1.1.1", "ingest") == 0
    assert rl.remaining("2.2.2.2", "ingest") == 1
