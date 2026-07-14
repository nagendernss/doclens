import threading
from datetime import datetime, timezone


class RateLimiter:
    """Thread-safe rate limiter with per-IP per-kind and global daily limits."""

    def __init__(self, per_ip_ingest, per_ip_question, global_cap, today=None):
        """Initialize rate limiter.

        Args:
            per_ip_ingest: Daily limit for ingest requests per IP.
            per_ip_question: Daily limit for question requests per IP.
            global_cap: Global daily limit across all IPs and kinds.
            today: Callable returning UTC date string (e.g., "2026-07-14").
                   Defaults to current UTC date.
        """
        self.per_ip_ingest = per_ip_ingest
        self.per_ip_question = per_ip_question
        self.global_cap = global_cap
        self.today = today or (lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        self.lock = threading.Lock()
        self._day = None
        # Structure: per_ip_counters[ip][kind] = count
        self.per_ip_counters = {}
        # Structure: global_counters = count
        self.global_counters = 0

    def _roll(self):
        """Reset counters if day has changed (must be called with lock held)."""
        today = self.today()
        if self._day != today:
            self._day = today
            self.per_ip_counters = {}
            self.global_counters = 0

    def allow(self, ip: str, kind: str) -> tuple[bool, str]:
        """Check if request is allowed and increment counters if allowed.

        Only counts towards limits if request is allowed (not on deny).

        Args:
            ip: Client IP address.
            kind: Request kind, either "ingest" or "question".

        Returns:
            Tuple of (allowed: bool, reason: str).
            On denial, reason contains "daily limit" for per-IP or "global" for global.
        """
        with self.lock:
            self._roll()

            # Initialize per-IP counter structure for this kind
            if ip not in self.per_ip_counters:
                self.per_ip_counters[ip] = {}
            if kind not in self.per_ip_counters[ip]:
                self.per_ip_counters[ip][kind] = 0

            # Get the per-IP limit for this kind
            limit = (
                self.per_ip_ingest if kind == "ingest" else self.per_ip_question
            )

            # Check per-IP limit
            if self.per_ip_counters[ip][kind] >= limit:
                return False, f"{kind} daily limit"

            # Check global limit
            if self.global_counters >= self.global_cap:
                return False, "global daily limit"

            # Allow and increment counters
            self.per_ip_counters[ip][kind] += 1
            self.global_counters += 1

            return True, f"{kind} allowed"

    def remaining(self, ip: str, kind: str) -> int:
        """Get remaining requests for this IP and kind today.

        Args:
            ip: Client IP address.
            kind: Request kind, either "ingest" or "question".

        Returns:
            Number of remaining requests before hitting per-IP limit.
        """
        with self.lock:
            self._roll()

            # Initialize per-IP counter structure for this kind
            if ip not in self.per_ip_counters:
                self.per_ip_counters[ip] = {}
            if kind not in self.per_ip_counters[ip]:
                self.per_ip_counters[ip][kind] = 0

            limit = (
                self.per_ip_ingest if kind == "ingest" else self.per_ip_question
            )
            current = self.per_ip_counters[ip][kind]

            return limit - current
