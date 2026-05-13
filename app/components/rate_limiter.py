"""In-process sliding-window rate limiter for LLM calls."""
from __future__ import annotations

import threading
import time
from collections import deque


class TokenBucketLimiter:
    def __init__(self, max_requests: int = 10, window_sec: float = 60.0) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if window_sec <= 0:
            raise ValueError("window_sec must be > 0")
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        now = time.monotonic()
        with self._lock:
            while self._timestamps and now - self._timestamps[0] > self.window_sec:
                self._timestamps.popleft()
            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return True
            return False

    def wait(self, poll_sec: float = 0.25) -> None:
        while not self.acquire():
            time.sleep(poll_sec)
