"""What sharing state through Redis changes when the gateway runs as several processes.

    BENCH_REDIS_URL=redis://localhost:6379/15 python -m bench.distributed     # writes bench/DISTRIBUTED.md

Starts three gateway processes on one machine and spreads requests across them round-robin, as a
load balancer would. Every scenario runs twice: each process keeping its own state, then all of
them sharing it through Redis. The Redis database named in BENCH_REDIS_URL is flushed between runs,
so point it at one you don't use for anything else.
"""

import asyncio
import contextlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from redis.asyncio import Redis

from bench.run import ROOT, _cpu_name, _free_port, _prepare_database, ms
from gateway.db import make_engine, make_sessions
from gateway.keys import create_key

PROCESSES = 3
MODES = {"each process alone": False, "shared through Redis": True}


@contextlib.asynccontextmanager
async def cluster(scenario: dict, processes: int = PROCESSES):
    """Start `processes` gateways sharing one database (and Redis, if the scenario says so)."""
    redis_url = os.environ["BENCH_REDIS_URL"]
    redis = Redis.from_url(redis_url)
    await redis.flushdb()
    await redis.aclose()
    with tempfile.TemporaryDirectory() as tmp:
        url = os.environ.get("BENCH_DATABASE_URL") or f"sqlite+aiosqlite:///{tmp}/bench.db"
        key = await _prepare_database(url)
        env = os.environ | {"BENCH_SCENARIO": json.dumps(scenario), "DATABASE_URL": url}
        ports = [_free_port() for _ in range(processes)]
        procs = [subprocess.Popen([sys.executable, "-m", "uvicorn", "--factory", "bench.app:create", "--port", str(p),
                                   "--log-level", "warning", "--no-access-log"],
                                  cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                 for p in ports]
        bases = [f"http://127.0.0.1:{p}" for p in ports]
        try:
            async with httpx.AsyncClient() as client:
                for base in bases:
                    for _ in range(100):
                        with contextlib.suppress(httpx.TransportError):
                            if (await client.get(f"{base}/healthz")).status_code == 200:
                                break
                        await asyncio.sleep(0.1)
                    else:
                        raise RuntimeError(f"{base} did not start")
            yield bases, key, url
        finally:
            for p in procs:
                p.terminate()
            for p in procs:
                p.wait(timeout=10)


async def new_key(url: str, rpm: int) -> str:
    engine = make_engine(url)
    try:
        async with make_sessions(engine)() as session:
            return (await create_key(session, "bench-limited", rpm=rpm))[1]
    finally:
        await engine.dispose()


async def spread(bases: list[str], key: str, bodies: list[dict], concurrency: int):
    """Send the bodies round-robin across the processes, `concurrency` at a time.
    Returns [(status, seconds, headers)] in send order, and the wall time."""
    sem = asyncio.Semaphore(concurrency)
    clients = [httpx.AsyncClient(base_url=b, timeout=60, headers={"authorization": f"Bearer {key}"},
                                 limits=httpx.Limits(max_connections=concurrency)) for b in bases]
    try:
        async def one(i: int, body: dict):
            async with sem:
                t = time.perf_counter()
                r = await clients[i % len(clients)].post("/v1/chat/completions", json=body)
                return r.status_code, time.perf_counter() - t, r.headers
        started = time.perf_counter()
        out = await asyncio.gather(*(one(i, b) for i, b in enumerate(bodies)))
        return out, time.perf_counter() - started
    finally:
        for c in clients:
            await c.aclose()


def ask(text: str, **extra) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": text}], **extra}


ONE_MODEL = {"m": [{"provider": "model", "model": "m-1"}]}


# --------------------------------------------------------------------------- scenarios
async def rate_limit() -> dict:
    """One key limited to 30 requests a minute sends 90 requests at once, spread over the processes."""
    out = {}
    for mode, shared in MODES.items():
        scenario = {"providers": [{"name": "model"}], "routes": ONE_MODEL, "redis": shared}
        async with cluster(scenario) as (bases, _, url):
            key = await new_key(url, rpm=30)
            results, _ = await spread(bases, key, [ask(f"q{i}") for i in range(90)], 9)
        out[mode] = {"limit_rpm": 30, "sent": 90, "allowed": sum(s == 200 for s, *_ in results),
                     "refused_429": sum(s == 429 for s, *_ in results)}
    return out


async def coalescing() -> dict:
    """60 identical temperature-0 requests arrive at once, 20 at each process; the provider takes 500 ms."""
    setups = {"cache only, each process alone": (False, False),
              "cache + coalescing, each process alone": (False, True),
              "cache + coalescing, shared through Redis": (True, True)}
    out = {}
    for name, (shared, coalesce) in setups.items():
        scenario = {"providers": [{"name": "model", "latency_s": 0.5}], "routes": ONE_MODEL, "cache": True,
                    "redis": shared, "coalesce": coalesce}
        async with cluster(scenario) as (bases, key, _):
            results, wall = await spread(bases, key, [ask("What is our refund window?", temperature=0)] * 60, 60)
        kinds = [h.get("x-gateway-cache") for _, _, h in results]
        out[name] = {"requests": 60, "ok": sum(s == 200 for s, *_ in results), "provider_calls": kinds.count("miss"),
                     "coalesced": kinds.count("coalesced"), "cache_hits": kinds.count("hit"),
                     "p50_ms": ms([t for _, t, _ in results], 50), "total_s": round(wall, 2)}
    return out


async def breakers() -> dict:
    """The first provider hangs (1 s timeout), the backup answers in 50 ms. 60 requests, one in flight
    per process. A breaker opens after 5 timeouts: per process, or once for everyone."""
    out = {}
    for mode, shared in MODES.items():
        scenario = {"providers": [{"name": "hung", "latency_s": 600}, {"name": "backup", "latency_s": 0.05}],
                    "routes": {"m": [{"provider": "hung", "model": "h"}, {"provider": "backup", "model": "b"}]},
                    "policy": {"attempt_timeout_s": 1.0, "breaker_threshold": 5, "breaker_cooldown_s": 300},
                    "redis": shared}
        async with cluster(scenario) as (bases, key, _):
            results, wall = await spread(bases, key, [ask(f"q{i}") for i in range(60)], PROCESSES)
        lat = [t for _, t, _ in results]
        out[mode] = {"requests": 60, "ok": sum(s == 200 for s, *_ in results),
                     "paid_the_timeout": sum(t >= 1.0 for t in lat), "mean_ms": round(statistics.mean(lat) * 1000, 1),
                     "total_s": round(wall, 1)}
    return out


async def overhead() -> dict:
    """What the Redis round trips cost: an instant provider, requests the cache can't answer."""
    out = {}
    for mode, shared in MODES.items():
        scenario = {"providers": [{"name": "model"}], "routes": ONE_MODEL, "redis": shared}
        row = {}
        async with cluster(scenario, processes=1) as (bases, key, _):
            await spread(bases, key, [ask("warm")] * 50, 5)
            results, _ = await spread(bases, key, [ask(f"q{i}") for i in range(500)], 1)
            row["p50_ms_one_process_one_at_a_time"] = ms([t for _, t, _ in results], 50)
        async with cluster(scenario) as (bases, key, _):
            await spread(bases, key, [ask("warm")] * 60, 15)
            results, wall = await spread(bases, key, [ask(f"q{i}") for i in range(1500)], 30)
            row["requests_per_s_three_processes"] = round(1500 / wall, 1)
            row["p50_ms_three_processes"] = ms([t for _, t, _ in results], 50)
        out[mode] = row
    return out


# --------------------------------------------------------------------------- report
def report(r: dict) -> str:
    m = r["machine"]
    lines = ["# Distributed benchmark", "",
             f"Measured {r['date']} on {m['cpus']} vCPU ({m['cpu']}), Redis {m['redis']}, ledger on {m['database']}. "
             f"{PROCESSES} gateway processes on one machine; requests are spread round-robin, as a load balancer would. "
             "Each scenario runs with every process keeping its own state, then with the state shared through Redis. "
             "Reproduce with `BENCH_REDIS_URL=redis://localhost:6379/15 python -m bench.distributed`.", "",
             "## One rate limit, three processes", "",
             "A key limited to 30 requests a minute sends 90 at once.", "",
             "| State | Allowed | Refused (429) |", "|---|---|---|"]
    for mode, x in r["rate_limit"].items():
        lines.append(f"| {mode} | {x['allowed']} of {x['sent']} | {x['refused_429']} |")
    lines += ["", "## Identical requests in flight", "",
              "60 identical temperature-0 requests arrive at once, 20 at each process; the provider takes 500 ms.", "",
              "| Setup | Provider calls | Answered by another request in flight | p50 |", "|---|---|---|---|"]
    for name, x in r["coalescing"].items():
        lines.append(f"| {name} | {x['provider_calls']} | {x['coalesced'] + x['cache_hits']} | {x['p50_ms']} ms |")
    lines += ["", "## A provider that hangs", "",
              "1 s timeout, backup answers in 50 ms, 60 requests with one in flight per process; breakers open after 5 timeouts.", "",
              "| State | Success | Requests that waited out the timeout | Mean | Total time |", "|---|---|---|---|---|"]
    for mode, x in r["breakers"].items():
        lines.append(f"| {mode} | {x['ok']} of {x['requests']} | {x['paid_the_timeout']} | {x['mean_ms']} ms | {x['total_s']} s |")
    lines += ["", "## What Redis costs", "",
              "An instant provider and requests the cache can't answer: each request makes a rate-limit call and two "
              "circuit-breaker calls to Redis.", "",
              "| State | p50, one process, one request at a time | p50, three processes, 30 in flight | Requests/s, three processes |",
              "|---|---|---|---|"]
    for mode, x in r["overhead"].items():
        lines.append(f"| {mode} | {x['p50_ms_one_process_one_at_a_time']} ms | {x['p50_ms_three_processes']} ms | "
                     f"{x['requests_per_s_three_processes']} |")
    lines.append("")
    return "\n".join(lines)


async def main() -> None:
    if not os.environ.get("BENCH_REDIS_URL"):
        sys.exit("Set BENCH_REDIS_URL to a Redis database this can flush, e.g. redis://localhost:6379/15")
    redis = Redis.from_url(os.environ["BENCH_REDIS_URL"])
    version = (await redis.info("server"))["redis_version"]
    await redis.aclose()
    results = {"date": time.strftime("%Y-%m-%d"),
               "machine": {"cpus": os.cpu_count(), "cpu": _cpu_name(), "redis": version,
                           "database": "Postgres" if os.environ.get("BENCH_DATABASE_URL") else "SQLite"}}
    for name, fn in (("rate_limit", rate_limit), ("coalescing", coalescing), ("breakers", breakers),
                     ("overhead", overhead)):
        print(f"running {name}...", flush=True)
        results[name] = await fn()
    out = Path(__file__).parent
    (out / "distributed.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "DISTRIBUTED.md").write_text(report(results))
    print(report(results))


if __name__ == "__main__":
    asyncio.run(main())
