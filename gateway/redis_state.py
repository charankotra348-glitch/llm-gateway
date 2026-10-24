"""State shared by every gateway process through Redis: rate-limit buckets, circuit breakers and the
response cache. With REDIS_URL unset, the in-process versions are used instead and each process
keeps its own (fine for one process; with several, a limit of N becomes N per process).

Each read-modify-write runs as a Lua script, which Redis executes atomically: two gateway processes
can't both take the last token or both claim a breaker's probe. Scripts use Redis's own clock
(TIME), so the processes' clocks don't need to agree.

If Redis is unreachable, the gateway keeps serving rather than failing every request: each
operation falls back to the process's local state (rate limits and breakers keep working per
process; the cache misses) and logs `redis_unavailable` at most once every 10 seconds.
"""

import json
import logging
import time
import uuid

from redis.asyncio import Redis
from redis.exceptions import RedisError

from gateway.cache import Entry
from gateway.limits import Decision, RateLimiter
from gateway.resilience import LocalBreakers, Permit
from gateway.routes import Target
from gateway.schemas import Completion

log = logging.getLogger("gateway")
PREFIX = "gw:"


class _Outages:
    """Logs that Redis is unreachable, once per 10 s rather than once per request."""

    def __init__(self):
        self._last = 0.0

    def report(self, what: str, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last >= 10:
            self._last = now
            log.warning(json.dumps({"event": "redis_unavailable", "during": what, "error": type(exc).__name__,
                                    "fallback": "local state"}))


_outages = _Outages()

# --------------------------------------------------------------------------- rate limits
TOKEN_BUCKET = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local rpm = tonumber(ARGV[1])
local rate = rpm / 60
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1]) or rpm
local ts = tonumber(state[2]) or now
tokens = math.min(rpm, tokens + math.max(0, now - ts) * rate)
local allowed, retry = 0, 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  retry = (1 - tokens) / rate
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('PEXPIRE', KEYS[1], math.ceil(120000))
return {allowed, math.floor(tokens), tostring(retry)}
"""


class RedisRateLimiter:
    """The token bucket from limits.py, with the bucket in Redis so every process draws from one."""

    def __init__(self, redis: Redis, default_rpm: int = 60):
        self.default_rpm = default_rpm
        self._redis = redis
        self._script = redis.register_script(TOKEN_BUCKET)
        self._fallback = RateLimiter(default_rpm)

    async def take(self, key_id: str, key_rpm: int | None) -> Decision | None:
        rpm = self.default_rpm if key_rpm is None else key_rpm
        if rpm <= 0:
            return None
        try:
            allowed, remaining, retry = await self._script(keys=[f"{PREFIX}rl:{key_id}:{rpm}"], args=[rpm])
        except RedisError as exc:
            _outages.report("rate_limit", exc)
            return await self._fallback.take(key_id, key_rpm)
        return Decision(bool(allowed), rpm, int(remaining), float(retry))


# --------------------------------------------------------------------------- response cache
class RedisResponseCache:
    """The response cache in Redis: one entry per key with a TTL. Least-recently-used eviction is
    Redis's job (maxmemory-policy allkeys-lru), so there's no entry limit here."""

    def __init__(self, redis: Redis, ttl_s: float = 3600):
        self.ttl_s = ttl_s
        self._redis = redis

    @property
    def enabled(self) -> bool:
        return self.ttl_s > 0

    async def aget(self, key: str) -> Entry | None:
        try:
            raw = await self._redis.get(f"{PREFIX}cache:{key}")
        except RedisError as exc:
            _outages.report("cache_read", exc)
            return None
        if raw is None:
            return None
        data = json.loads(raw)
        return Completion(**data["completion"]), Target(**data["target"])

    async def aput(self, key: str, value: Entry) -> None:
        completion, target = value
        payload = json.dumps({"completion": completion.model_dump(), "target": target.model_dump()})
        try:
            await self._redis.set(f"{PREFIX}cache:{key}", payload, px=int(self.ttl_s * 1000))
        except RedisError as exc:
            _outages.report("cache_write", exc)


# --------------------------------------------------------------------------- circuit breakers
# State per target: a hash {failures, opened_at (ms, 0 = closed)} and a probe key holding the token
# of the one request allowed through a half-open breaker.
BREAKER_ALLOW = """
local opened = tonumber(redis.call('HGET', KEYS[1], 'opened_at') or '0')
if opened == 0 then return 1 end
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
if now - opened < tonumber(ARGV[1]) then return 0 end
if redis.call('SET', KEYS[2], ARGV[2], 'NX', 'PX', ARGV[3]) then return 2 end
return 0
"""

BREAKER_FAILURE = """
local failures = redis.call('HINCRBY', KEYS[1], 'failures', 1)
local opened = tonumber(redis.call('HGET', KEYS[1], 'opened_at') or '0')
local probing = ARGV[2] ~= '' and redis.call('GET', KEYS[2]) == ARGV[2]
if probing or (opened == 0 and failures >= tonumber(ARGV[1])) then
  local t = redis.call('TIME')
  redis.call('HSET', KEYS[1], 'opened_at', tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000))
  if probing then redis.call('DEL', KEYS[2]) end
end
redis.call('PEXPIRE', KEYS[1], ARGV[3])
return failures
"""

BREAKER_SUCCESS = """
redis.call('DEL', KEYS[1])
if ARGV[1] ~= '' and redis.call('GET', KEYS[2]) == ARGV[1] then redis.call('DEL', KEYS[2]) end
return 1
"""

BREAKER_RELEASE = """
if ARGV[1] ~= '' and redis.call('GET', KEYS[1]) == ARGV[1] then redis.call('DEL', KEYS[1]) end
return 1
"""


class RedisBreakers:
    """Circuit breakers shared by every process: once any of them sees `threshold` failures in a row
    from a target, all of them skip it, and only one request across all of them probes it after the
    cooldown. Same states and rules as resilience.CircuitBreaker."""

    def __init__(self, redis: Redis, threshold: int = 5, cooldown_s: float = 30.0, probe_ttl_s: float = 60.0):
        self.threshold, self.cooldown_s = threshold, cooldown_s
        self._redis = redis
        self._probe_ttl_ms = int(probe_ttl_s * 1000)        # a probe whose process died frees itself
        self._forget_ms = int(max(cooldown_s * 10, 600) * 1000)
        self._allow = redis.register_script(BREAKER_ALLOW)
        self._failure = redis.register_script(BREAKER_FAILURE)
        self._success = redis.register_script(BREAKER_SUCCESS)
        self._release = redis.register_script(BREAKER_RELEASE)
        self._fallback = LocalBreakers(threshold, cooldown_s)

    @staticmethod
    def _keys(target: Target) -> list[str]:
        base = f"{PREFIX}breaker:{target.provider}:{target.model}"
        return [base, f"{base}:probe"]

    async def allow(self, target: Target) -> Permit | None:
        token = uuid.uuid4().hex
        try:
            verdict = await self._allow(keys=self._keys(target),
                                        args=[int(self.cooldown_s * 1000), token, self._probe_ttl_ms])
        except RedisError as exc:
            _outages.report("breaker", exc)
            return await self._fallback.allow(target)
        if verdict == 0:
            return None
        return Permit(probe=verdict == 2, token=token if verdict == 2 else None)

    async def success(self, target: Target, permit: Permit) -> None:
        await self._run(self._success, self._keys(target), [permit.token or ""])

    async def failure(self, target: Target, permit: Permit) -> None:
        await self._run(self._failure, self._keys(target), [self.threshold, permit.token or "", self._forget_ms])

    async def release(self, target: Target, permit: Permit) -> None:
        if permit.token:
            await self._run(self._release, self._keys(target)[1:], [permit.token])

    async def _run(self, script, keys: list[str], args: list) -> None:
        try:
            await script(keys=keys, args=args)
        except RedisError as exc:
            _outages.report("breaker", exc)

    async def states(self) -> list[dict]:
        out = []
        try:
            async for key in self._redis.scan_iter(match=f"{PREFIX}breaker:*"):
                key = key.decode() if isinstance(key, bytes) else key
                if key.endswith(":probe"):
                    continue
                _, _, provider, model = key.split(":", 3)
                data = await self._redis.hgetall(key)
                data = {(k.decode() if isinstance(k, bytes) else k): int(v) for k, v in data.items()}
                opened = data.get("opened_at", 0)
                state = "closed"
                if opened:
                    now_s, now_us = await self._redis.time()
                    elapsed = now_s * 1000 + now_us // 1000 - opened
                    state = "open" if elapsed < self.cooldown_s * 1000 else "half_open"
                out.append({"provider": provider, "model": model, "state": state,
                            "consecutive_failures": data.get("failures", 0)})
        except RedisError as exc:
            _outages.report("breaker_states", exc)
        return sorted(out, key=lambda s: (s["provider"], s["model"]))
