"""Request coalescing ("single-flight"): identical cacheable requests that arrive while the first is
still waiting on a provider don't each call the provider. One request (the leader) calls it; the
others wait and are answered with its result.

Only requests the cache may answer are coalesced (temperature 0, same API key and route, identical
options), so a follower receives exactly what a cache hit a moment later would have given it.

Two layers:
- Within one process, followers await the leader's asyncio future. No I/O.
- Across processes (with Redis), the leader holds a short lock, `SET NX PX`; followers in other
  processes poll the shared cache until the answer appears, or until the lock disappears without
  one (the leader failed), in which case one of them takes the lock and becomes the new leader.

A leader's failure is shared with its local followers: they asked the same question of the same
providers a moment later and would very likely get the same error.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

from gateway.cache import Entry

log = logging.getLogger("gateway")

RELEASE = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end return 0"


@dataclass
class Joined:
    """A follower's answer, produced by another request."""

    value: Entry


class Lead:
    """This request calls the provider. Finish with done() or failed(), then release() (in finally)."""

    def __init__(self, coalescer: "Coalescer", key: str, future: asyncio.Future, token: str | None):
        self._c, self._key, self._future, self._token = coalescer, key, future, token

    def done(self, value: Entry) -> None:
        if not self._future.done():
            self._future.set_result(value)

    def failed(self, exc: BaseException) -> None:
        if not self._future.done():
            self._future.set_exception(exc)
            self._future.exception()            # mark retrieved: no "never retrieved" warning without followers

    async def release(self) -> None:
        if not self._future.done():             # left without an answer (cancelled): let followers retry
            self._future.cancel()
        if self._c._local.get(self._key) is self._future:
            del self._c._local[self._key]       # the next identical request starts afresh (or hits the cache)
        if self._token and self._c._redis is not None:
            try:
                await self._c._release(keys=[self._c._lock_key(self._key)], args=[self._token])
            except RedisError:
                pass                            # the lock expires by itself


class Coalescer:
    def __init__(self, redis: Redis | None = None, *, poll_s: float = 0.05, enabled: bool = True):
        self.enabled = enabled
        self._redis = redis
        self._poll_s = poll_s
        self._local: dict[str, asyncio.Future] = {}
        self._release = redis.register_script(RELEASE) if redis is not None else None
        self.joined = 0                          # followers answered, for tests and benchmarks

    @staticmethod
    def _lock_key(key: str) -> str:
        return f"gw:inflight:{key}"

    async def join(self, key: str, cache, timeout_s: float) -> Joined | Lead:
        """Become the leader for `key`, or wait for the current one's answer."""
        deadline = time.monotonic() + timeout_s
        while True:
            # 1. Same process: someone is already asking.
            if (future := self._local.get(key)) is not None:
                try:
                    value = await asyncio.wait_for(asyncio.shield(future), max(0.0, deadline - time.monotonic()))
                except (asyncio.CancelledError, TimeoutError):
                    if asyncio.current_task().cancelling():
                        raise
                    if time.monotonic() >= deadline:
                        return self._lead(key, None)    # waited long enough: ask ourselves
                    continue                            # the leader gave up; try again
                self.joined += 1
                return Joined(value)

            # 2. Nobody here: claim the key in this process, then across processes.
            future = asyncio.get_running_loop().create_future()
            self._local[key] = future
            if self._redis is None:
                return Lead(self, key, future, None)
            token = uuid.uuid4().hex
            try:
                got = await self._redis.set(self._lock_key(key), token, nx=True,
                                            px=max(1, int((deadline - time.monotonic()) * 1000)))
            except RedisError:
                return Lead(self, key, future, None)    # no Redis: coalesce within this process only
            if got:
                return Lead(self, key, future, token)

            # 3. Another process leads: wait for its answer in the shared cache.
            local_followers = Lead(self, key, future, None)   # local requests wait on us meanwhile
            try:
                while time.monotonic() < deadline:
                    await asyncio.sleep(self._poll_s)
                    if (hit := await cache.aget(key)) is not None:
                        local_followers.done(hit)
                        await local_followers.release()
                        self.joined += 1
                        return Joined(hit)
                    if not await self._redis.exists(self._lock_key(key)):
                        break                           # the leader finished without an answer
            except RedisError:
                pass
            await local_followers.release()
            if time.monotonic() >= deadline:
                return self._lead(key, None)
            # loop: try to become the leader ourselves

    def _lead(self, key: str, token: str | None) -> Lead:
        future = asyncio.get_running_loop().create_future()
        return Lead(self, f"{key}#solo-{uuid.uuid4().hex}", future, token)     # not shared with anyone
