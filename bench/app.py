"""App factory for the benchmarks: the real gateway (auth, ledger, rate limiter, cache, redaction,
resilience), with in-process fake providers shaped by the BENCH_SCENARIO environment variable.

    uvicorn --factory bench.app:create      # run by bench/run.py, not by hand
"""

import json
import os

from gateway.cache import ResponseCache
from gateway.coalesce import Coalescer
from gateway.db import make_engine, make_sessions
from gateway.keys import KeyCache
from gateway.ledger import LedgerWriter
from gateway.limits import RateLimiter
from gateway.main import Gateway, create_app
from gateway.pricing import PriceTable
from gateway.providers.fake import FakeProvider
from gateway.resilience import Resilience, ResiliencePolicy
from gateway.routes import RouteTable


def create():
    scenario = json.loads(os.environ["BENCH_SCENARIO"])
    providers = {
        p["name"]: FakeProvider(latency_s=p.get("latency_s", 0.0), failure_rate=p.get("failure_rate", 0.0),
                                seed=p.get("seed", 1))
        for p in scenario["providers"]
    }
    policy = ResiliencePolicy(**scenario.get("policy", {}))
    engine = make_engine(os.environ["DATABASE_URL"])
    sessions = make_sessions(engine)
    fast = scenario.get("fast_path", True)      # the production settings; False = a database trip per request
    ttl = 3600 if scenario.get("cache") else 0
    coalesce = scenario.get("coalesce", True)
    redis = None
    if scenario.get("redis"):                   # state shared by every process through Redis
        from redis.asyncio import Redis

        from gateway.redis_state import RedisBreakers, RedisRateLimiter, RedisResponseCache
        redis = Redis.from_url(os.environ["BENCH_REDIS_URL"])
        state = dict(resilience=Resilience(policy, breakers=RedisBreakers(redis, policy.breaker_threshold,
                                                                          policy.breaker_cooldown_s)),
                     limiter=RedisRateLimiter(redis, default_rpm=0), cache=RedisResponseCache(redis, ttl_s=ttl))
    else:                                       # each process keeps its own
        state = dict(resilience=Resilience(policy), limiter=RateLimiter(default_rpm=0), cache=ResponseCache(ttl_s=ttl))
    gateway = Gateway(
        providers, RouteTable.from_dict({"routes": scenario["routes"]}), sessions,
        prices=PriceTable.from_dict({"free_providers": list(providers)}), redact_pii=True,
        keys=KeyCache(ttl_s=30 if fast else 0), ledger=LedgerWriter(sessions, batched=fast),
        coalescer=Coalescer(redis, enabled=coalesce), redis=redis, engine=engine, **state)
    return create_app(gateway)
