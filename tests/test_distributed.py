"""Shared state across gateway processes, through Redis.

Two Gateway apps that share one Redis stand in for two server processes. Tests use fakeredis (an
in-memory Redis that runs the same Lua scripts) unless TEST_REDIS_URL points at a real one; CI
runs them against Redis 7.
"""

import asyncio
import json
import os

import fakeredis
import httpx
import pytest
from redis.asyncio import Redis

from gateway.cache import ResponseCache
from gateway.coalesce import Coalescer, Joined, Lead
from gateway.db import make_engine, make_sessions
from gateway.errors import ProviderTimeout, ProviderUnavailable
from gateway.keys import create_key
from gateway.limits import RateLimiter
from gateway.main import Gateway, create_app
from gateway.pricing import PriceTable
from gateway.providers.fake import FakeProvider
from gateway.redis_state import RedisBreakers, RedisRateLimiter, RedisResponseCache
from gateway.resilience import Resilience, ResiliencePolicy
from gateway.routes import RouteTable, Target
from gateway.schemas import Completion

pytestmark = pytest.mark.anyio
TARGET = Target(provider="p", model="m")
ASK = {"model": "smart", "temperature": 0, "messages": [{"role": "user", "content": "what is a refund window?"}]}


@pytest.fixture
async def redis():
    url = os.environ.get("TEST_REDIS_URL")
    client = Redis.from_url(url) if url else fakeredis.FakeAsyncRedis()
    await client.flushdb()
    yield client
    await client.aclose()


# --------------------------------------------------------------------------- rate limits
async def test_one_limit_shared_by_every_process(redis):
    a, b = RedisRateLimiter(redis, 0), RedisRateLimiter(redis, 0)
    results = [(await (a if i % 2 else b).take("key", 4)).allowed for i in range(8)]
    assert results.count(True) == 4                     # not 4 per process
    denied = await a.take("key", 4)
    assert not denied.allowed and 14 < denied.retry_after_s <= 15    # 60 s / 4 per token


async def test_redis_limits_are_per_key_and_follow_limit_changes(redis):
    limiter = RedisRateLimiter(redis, default_rpm=1)
    assert (await limiter.take("a", None)).allowed and not (await limiter.take("a", None)).allowed
    assert (await limiter.take("b", None)).allowed
    assert (await limiter.take("a", 5)).allowed          # a new limit starts a fresh bucket
    assert await limiter.take("a", 0) is None


# --------------------------------------------------------------------------- circuit breakers
async def test_failures_seen_by_one_process_open_the_breaker_for_all(redis):
    a, b = RedisBreakers(redis, threshold=3, cooldown_s=30), RedisBreakers(redis, threshold=3, cooldown_s=30)
    for store in (a, b, a):
        await store.failure(TARGET, await store.allow(TARGET))
    assert await b.allow(TARGET) is None and await a.allow(TARGET) is None
    [state] = await a.states()
    assert (state["state"], state["consecutive_failures"]) == ("open", 3)


async def test_one_probe_across_all_processes_after_the_cooldown(redis):
    a, b = RedisBreakers(redis, threshold=1, cooldown_s=0.2), RedisBreakers(redis, threshold=1, cooldown_s=0.2)
    await a.failure(TARGET, await a.allow(TARGET))
    await asyncio.sleep(0.25)
    probe = await b.allow(TARGET)
    assert probe is not None and probe.probe
    assert await a.allow(TARGET) is None                # the other process waits for the probe
    await b.success(TARGET, probe)
    assert (await a.allow(TARGET)) is not None and (await a.states()) == []


async def test_a_failed_probe_reopens_and_a_released_one_frees_the_slot(redis):
    a, b = RedisBreakers(redis, threshold=1, cooldown_s=0.2), RedisBreakers(redis, threshold=1, cooldown_s=0.2)
    await a.failure(TARGET, await a.allow(TARGET))
    await asyncio.sleep(0.25)
    probe = await a.allow(TARGET)
    await a.release(TARGET, probe)                       # e.g. the probe was a bad request
    probe = await b.allow(TARGET)
    assert probe is not None
    await b.failure(TARGET, probe)
    assert await a.allow(TARGET) is None                 # open again for a full cooldown


async def test_a_success_resets_the_shared_count(redis):
    store = RedisBreakers(redis, threshold=2, cooldown_s=30)
    await store.failure(TARGET, await store.allow(TARGET))
    await store.success(TARGET, await store.allow(TARGET))
    await store.failure(TARGET, await store.allow(TARGET))
    assert await store.allow(TARGET) is not None


async def test_model_names_with_colons(redis):
    store = RedisBreakers(redis, threshold=1, cooldown_s=30)
    nova = Target(provider="bedrock", model="us.amazon.nova-2-lite-v1:0")
    await store.failure(nova, await store.allow(nova))
    [state] = await store.states()
    assert (state["provider"], state["model"]) == ("bedrock", "us.amazon.nova-2-lite-v1:0")


# --------------------------------------------------------------------------- cache
async def test_redis_cache_round_trip_and_ttl(redis):
    cache = RedisResponseCache(redis, ttl_s=0.2)
    entry = (Completion(text="hi", model="m", input_tokens=3, output_tokens=1), TARGET)
    await cache.aput("k", entry)
    assert await RedisResponseCache(redis).aget("k") == entry      # another process reads it
    await asyncio.sleep(0.3)
    assert await cache.aget("k") is None


# --------------------------------------------------------------------------- Redis down
async def test_redis_outage_falls_back_to_local_state(caplog):
    dead = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)
    try:
        limiter = RedisRateLimiter(dead, default_rpm=2)
        assert [(await limiter.take("k", None)).allowed for _ in range(3)] == [True, True, False]   # per process
        assert await RedisResponseCache(dead).aget("k") is None
        await RedisResponseCache(dead).aput("k", (Completion(text="x", model="m"), TARGET))
        breakers = RedisBreakers(dead, threshold=1)
        assert await breakers.allow(TARGET) is not None
        assert caplog.text.count("redis_unavailable") == 1      # logged once, not per call
    finally:
        await dead.aclose()


# --------------------------------------------------------------------------- coalescing, unit
async def test_local_followers_get_the_leaders_answer():
    c, cache = Coalescer(), ResponseCache(ttl_s=60)
    lead = await c.join("k", cache, 5)
    assert isinstance(lead, Lead)
    followers = [asyncio.create_task(c.join("k", cache, 5)) for _ in range(3)]
    await asyncio.sleep(0.01)
    entry = (Completion(text="answer", model="m"), TARGET)
    lead.done(entry)
    await lead.release()
    assert [f.value for f in await asyncio.gather(*followers)] == [entry] * 3
    assert isinstance(await c.join("k", cache, 5), Lead)       # finished: the next request starts afresh


async def test_local_followers_share_the_leaders_error():
    c, cache = Coalescer(), ResponseCache(ttl_s=60)
    lead = await c.join("k", cache, 5)
    follower = asyncio.create_task(c.join("k", cache, 5))
    await asyncio.sleep(0.01)
    lead.failed(ProviderUnavailable("down"))
    await lead.release()
    with pytest.raises(ProviderUnavailable):
        await follower


async def test_a_cancelled_leader_hands_over():
    c, cache = Coalescer(), ResponseCache(ttl_s=60)
    lead = await c.join("k", cache, 5)
    follower = asyncio.create_task(c.join("k", cache, 5))
    await asyncio.sleep(0.01)
    await lead.release()                                         # left without an answer
    assert isinstance(await follower, Lead)


async def test_followers_in_another_process_wait_for_the_shared_cache(redis):
    cache = RedisResponseCache(redis)
    a, b = Coalescer(redis, poll_s=0.01), Coalescer(redis, poll_s=0.01)
    lead = await a.join("k", cache, 5)
    remote = asyncio.create_task(b.join("k", cache, 5))
    await asyncio.sleep(0.05)
    assert not remote.done()
    entry = (Completion(text="answer", model="m"), TARGET)
    await cache.aput("k", entry)
    lead.done(entry)
    await lead.release()
    joined = await remote
    assert isinstance(joined, Joined) and joined.value == entry


async def test_another_process_takes_over_when_the_leader_fails(redis):
    cache = RedisResponseCache(redis)
    a, b = Coalescer(redis, poll_s=0.01), Coalescer(redis, poll_s=0.01)
    lead = await a.join("k", cache, 5)
    remote = asyncio.create_task(b.join("k", cache, 5))
    await asyncio.sleep(0.05)
    lead.failed(ProviderTimeout("slow"))
    await lead.release()                                         # no answer in the cache
    assert isinstance(await remote, Lead)


# --------------------------------------------------------------------------- two gateway processes
async def _process(database_url: str, redis: Redis, provider: FakeProvider, *, breaker_threshold: int = 5):
    policy = ResiliencePolicy(backoff_s=0, attempt_timeout_s=1.0, breaker_threshold=breaker_threshold)
    engine = make_engine(database_url)
    gw = Gateway(
        {"primary": provider, "backup": FakeProvider()},
        RouteTable.from_dict({"routes": {"smart": [{"provider": "primary", "model": "p-1"},
                                                   {"provider": "backup", "model": "b-1"}]}}),
        make_sessions(engine), prices=PriceTable.from_dict({"free_providers": ["primary", "backup"]}),
        resilience=Resilience(policy, breakers=RedisBreakers(redis, breaker_threshold, 30)),
        limiter=RedisRateLimiter(redis, default_rpm=0), cache=RedisResponseCache(redis, 60),
        coalescer=Coalescer(redis, poll_s=0.01), engine=engine)
    app = create_app(gw)
    return app, gw


@pytest.fixture
async def two_processes(database_url, redis):
    """Two gateway apps sharing Redis and the database; returns (client_a, client_b, provider_a, provider_b, key)."""
    pa, pb = FakeProvider(latency_s=0.2), FakeProvider(latency_s=0.2)
    (app_a, _), (app_b, _) = await _process(database_url, redis, pa), await _process(database_url, redis, pb)
    engine = make_engine(database_url)
    async with make_sessions(engine)() as s:
        _, key = await create_key(s, "two", rpm=0)
    await engine.dispose()
    headers = {"authorization": f"Bearer {key}"}
    async with app_a.router.lifespan_context(app_a), app_b.router.lifespan_context(app_b):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_a), base_url="http://a",
                                     headers=headers) as a, \
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app_b), base_url="http://b",
                                  headers=headers) as b:
            yield a, b, pa, pb, key


async def test_identical_requests_across_processes_reach_the_provider_once(two_processes):
    a, b, pa, pb, _ = two_processes
    responses = await asyncio.gather(*(c.post("/v1/chat/completions", json=ASK) for c in [a, b] * 5))
    assert all(r.status_code == 200 for r in responses)
    assert len(pa.calls) + len(pb.calls) == 1
    kinds = sorted(r.headers["x-gateway-cache"] for r in responses)
    assert kinds.count("miss") == 1 and kinds.count("coalesced") + kinds.count("hit") == 9
    assert len({json.dumps(r.json()["choices"]) for r in responses}) == 1


async def test_a_rate_limit_holds_across_processes(database_url, redis, two_processes):
    a, b, *_ = two_processes
    engine = make_engine(database_url)
    async with make_sessions(engine)() as s:
        _, key = await create_key(s, "limited", rpm=4)
    await engine.dispose()
    h = {"authorization": f"Bearer {key}"}
    body = {"model": "smart", "messages": [{"role": "user", "content": "hi"}]}
    statuses = [(await c.post("/v1/chat/completions", json=body, headers=h)).status_code for c in [a, b] * 4]
    assert statuses.count(200) == 4 and statuses.count(429) == 4


async def test_a_breaker_opened_by_one_process_protects_the_other(two_processes):
    a, b, pa, pb, _ = two_processes
    pa.script = [ProviderTimeout("hung")] * 5
    body = {"model": "smart", "messages": [{"role": "user", "content": "hi"}]}
    for i in range(5):                                    # five timeouts, all seen by process A
        r = await a.post("/v1/chat/completions", json=body | {"messages": [{"role": "user", "content": str(i)}]})
        assert r.headers["x-gateway-provider"] == "backup"
    r = await b.post("/v1/chat/completions", json=body)
    assert r.headers["x-gateway-provider"] == "backup" and pb.calls == []    # B skipped primary without trying
    status = (await b.get("/v1/status")).json()["targets"]
    assert {(s["provider"], s["state"]) for s in status} >= {("primary", "open")}


async def test_local_mode_still_coalesces_within_one_process(database_url, make_key):
    provider = FakeProvider(latency_s=0.2)
    engine = make_engine(database_url)
    gw = Gateway({"primary": provider}, RouteTable.from_dict({"routes": {"smart": [{"provider": "primary",
                                                                                   "model": "p-1"}]}}),
                 make_sessions(engine), prices=PriceTable.from_dict({"free_providers": ["primary"]}),
                 limiter=RateLimiter(0), cache=ResponseCache(ttl_s=60), engine=engine)
    app = create_app(gw)
    async with make_sessions(engine)() as s:
        _, key = await create_key(s, "solo")
    async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://x",
            headers={"authorization": f"Bearer {key}"}) as c:
        responses = await asyncio.gather(*(c.post("/v1/chat/completions", json=ASK) for _ in range(8)))
    assert all(r.status_code == 200 for r in responses) and len(provider.calls) == 1
    assert sorted(r.headers["x-gateway-cache"] for r in responses).count("coalesced") == 7
