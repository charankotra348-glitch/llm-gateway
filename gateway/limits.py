"""Per-key rate limits: a token bucket per key.

A bucket holds up to `rpm` tokens and refills at rpm/60 tokens a second. Each request takes one.
A full bucket allows a short burst of `rpm` requests; after that, requests are spaced by the refill
rate. That is the usual shape of API rate limits, OpenAI's included.

State lives in this process: with several replicas, each enforces the limit separately, so the
effective limit is rpm x replicas. Shared state (Redis) would make it exact.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_s: float = 0.0      # how long until one token is back, when not allowed


class TokenBucket:
    def __init__(self, rpm: int, clock: Callable[[], float] = time.monotonic):
        self.rpm = rpm
        self._clock = clock
        self._tokens = float(rpm)
        self._updated = clock()

    def take(self) -> Decision:
        now = self._clock()
        rate = self.rpm / 60.0
        self._tokens = min(self.rpm, self._tokens + (now - self._updated) * rate)
        self._updated = now
        if self._tokens >= 1:
            self._tokens -= 1
            return Decision(True, self.rpm, math.floor(self._tokens))
        return Decision(False, self.rpm, 0, (1 - self._tokens) / rate)


class RateLimiter:
    def __init__(self, default_rpm: int = 60, clock: Callable[[], float] = time.monotonic):
        self.default_rpm = default_rpm       # for keys without their own limit; 0 = unlimited
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}

    def check(self, key_id: str, key_rpm: int | None) -> Decision | None:
        """None when the key is unlimited."""
        rpm = self.default_rpm if key_rpm is None else key_rpm
        if rpm <= 0:
            return None
        bucket = self._buckets.get(key_id)
        if bucket is None or bucket.rpm != rpm:          # new key, or its limit was changed
            bucket = self._buckets[key_id] = TokenBucket(rpm, self._clock)
        return bucket.take()
