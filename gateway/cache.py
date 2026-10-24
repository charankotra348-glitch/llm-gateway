"""Exact-match response cache.

Only requests with temperature 0 are cached. With any other temperature the client asked for a
sample, and replaying the same sample would quietly change what the request means.

Entries are scoped to the API key, so one key can never receive an answer to another key's
prompt, or learn from response timing that someone else asked it.

Clients can steer it with standard HTTP Cache-Control:
    no-cache   don't answer from the cache (still store the fresh answer)
    no-store   don't read or write the cache

In-process LRU with a TTL. Several replicas each keep their own; with REDIS_URL set,
redis_state.RedisResponseCache shares one.
"""

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from gateway.routes import Target
from gateway.schemas import ChatRequest, Completion

Entry = tuple[Completion, Target]       # the answer, and the target that produced it


@dataclass
class CachePlan:
    key: str | None          # None: this request is not cacheable
    read: bool
    write: bool
    reason: str              # for the x-gateway-cache header when not a hit/miss: "bypass" | "no-cache" | "no-store"


def cache_key(api_key_id: str, route: str, chat: ChatRequest) -> str:
    payload = json.dumps({"key": api_key_id, "route": route, "request": chat.model_dump()}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def plan(api_key_id: str, route: str, chat: ChatRequest, cache_control: str | None) -> CachePlan:
    directives = {d.strip().lower() for d in (cache_control or "").split(",")}
    if chat.temperature != 0:
        return CachePlan(None, False, False, "bypass")
    if "no-store" in directives:
        return CachePlan(None, False, False, "no-store")
    key = cache_key(api_key_id, route, chat)
    if "no-cache" in directives:
        return CachePlan(key, False, True, "no-cache")
    return CachePlan(key, True, True, "")


class ResponseCache:
    def __init__(self, ttl_s: float = 3600, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, Entry]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.ttl_s > 0 and self.max_entries > 0

    def get(self, key: str) -> Entry | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if self._clock() - stored_at >= self.ttl_s:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)          # most recently used
        return value

    def put(self, key: str, value: Entry) -> None:
        if not self.enabled:
            return
        self._entries[key] = (self._clock(), value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)   # least recently used

    async def aget(self, key: str) -> Entry | None:
        """The interface main.py uses; shared with redis_state.RedisResponseCache."""
        return self.get(key)

    async def aput(self, key: str, value: Entry) -> None:
        self.put(key, value)

    def __len__(self) -> int:
        return len(self._entries)
