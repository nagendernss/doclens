import threading
from datetime import datetime


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
        self.today = today or (lambda: datetime.utcnow().strftime("%Y-%m-%d"))

        self.lock = threading.Lock()
        # Structure: per_ip_counters[ip][kind][date] = count
        self.per_ip_counters = {}
        # Structure: global_counters[date] = count
        self.global_counters = {}

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
            today = self.today()

            # Initialize per-IP counter structure for this kind
            if ip not in self.per_ip_counters:
                self.per_ip_counters[ip] = {}
            if kind not in self.per_ip_counters[ip]:
                self.per_ip_counters[ip][kind] = {}

            # Reset per-IP counter if date changed
            if today not in self.per_ip_counters[ip][kind]:
                self.per_ip_counters[ip][kind][today] = 0

            # Get the per-IP limit for this kind
            limit = (
                self.per_ip_ingest if kind == "ingest" else self.per_ip_question
            )

            # Check per-IP limit
            if self.per_ip_counters[ip][kind][today] >= limit:
                return False, f"{kind} daily limit"

            # Initialize global counter
            if today not in self.global_counters:
                self.global_counters[today] = 0

            # Check global limit
            if self.global_counters[today] >= self.global_cap:
                return False, "global daily limit"

            # Allow and increment counters
            self.per_ip_counters[ip][kind][today] += 1
            self.global_counters[today] += 1

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
            today = self.today()

            # Initialize per-IP counter structure for this kind
            if ip not in self.per_ip_counters:
                self.per_ip_counters[ip] = {}
            if kind not in self.per_ip_counters[ip]:
                self.per_ip_counters[ip][kind] = {}

            if today not in self.per_ip_counters[ip][kind]:
                self.per_ip_counters[ip][kind][today] = 0

            limit = (
                self.per_ip_ingest if kind == "ingest" else self.per_ip_question
            )
            current = self.per_ip_counters[ip][kind][today]

            return limit - current
