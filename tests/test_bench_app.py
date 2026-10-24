"""The benchmark app factory builds a working gateway from a scenario description."""

import asyncio
import json

from fastapi.testclient import TestClient

from bench.app import create
from gateway.db import make_engine, make_sessions
from gateway.keys import create_key


def test_bench_app_serves_a_scenario(database_url, monkeypatch):
    scenario = {"providers": [{"name": "model", "latency_s": 0}, {"name": "down", "failure_rate": 1.0}],
                "routes": {"m": [{"provider": "down", "model": "d"}, {"provider": "model", "model": "m-1"}]},
                "policy": {"backoff_s": 0}, "cache": True}
    monkeypatch.setenv("BENCH_SCENARIO", json.dumps(scenario))
    monkeypatch.setenv("DATABASE_URL", database_url)

    async def key():
        engine = make_engine(database_url)
        async with make_sessions(engine)() as s:
            _, k = await create_key(s, "bench", rpm=0)
        await engine.dispose()
        return k
    k = asyncio.run(key())
    with TestClient(create()) as client:
        r = client.post("/v1/chat/completions", headers={"authorization": f"Bearer {k}"},
                        json={"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200 and r.headers["x-gateway-provider"] == "model"
        assert r.headers["x-gateway-cache"] == "miss"
