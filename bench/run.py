"""Benchmarks: what the gateway costs in latency, and what it buys when providers misbehave.

    python -m bench.run                 # all four, about 4 minutes; writes bench/RESULTS.md and results.json
    python -m bench.run --quick         # smaller runs, for a smoke test
    BENCH_DATABASE_URL=postgresql+asyncpg://... python -m bench.run   # ledger on Postgres instead of SQLite

Each scenario starts the real gateway in its own uvicorn process (one worker), with fake providers
whose latency and failure rate are set per scenario, and drives it over HTTP from this process.
Providers are fakes on purpose: the point is to measure the gateway, not a model's speed or a
network, and to make failures happen on demand.
"""

import argparse
import asyncio
import contextlib
import json
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from alembic import command
from alembic.config import Config

from gateway.db import make_engine, make_sessions
from gateway.keys import create_key

ROOT = Path(__file__).resolve().parent.parent
QUESTION = [{"role": "user", "content": "Summarize our refund policy for a customer in two sentences."}]


# --------------------------------------------------------------------------- running the gateway
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _prepare_database(url: str) -> str:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes["database_url"] = url
    cfg.attributes["configure_logger"] = False
    await asyncio.to_thread(command.upgrade, cfg, "head")
    engine = make_engine(url)
    try:
        async with make_sessions(engine)() as session:
            _, key = await create_key(session, "bench", rpm=0)
    finally:
        await engine.dispose()
    return key


@contextlib.asynccontextmanager
async def gateway(scenario: dict):
    """Start the gateway for one scenario; yields (base_url, api_key)."""
    with tempfile.TemporaryDirectory() as tmp:
        url = os.environ.get("BENCH_DATABASE_URL") or f"sqlite+aiosqlite:///{tmp}/bench.db"
        key = await _prepare_database(url)
        port = _free_port()
        env = os.environ | {"BENCH_SCENARIO": json.dumps(scenario), "DATABASE_URL": url}
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "--factory", "bench.app:create", "--port", str(port),
             "--log-level", "warning", "--no-access-log"],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient() as client:
                for _ in range(100):
                    with contextlib.suppress(httpx.TransportError):
                        if (await client.get(f"{base}/healthz")).status_code == 200:
                            break
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("gateway did not start")
            yield base, key
        finally:
            proc.terminate()
            proc.wait(timeout=10)


@dataclass
class Sample:
    status: int
    seconds: float
    provider: str | None
    cache: str | None


async def fire(base: str, key: str, bodies: list[dict], concurrency: int) -> tuple[list[Sample], float]:
    """Send every body with at most `concurrency` in flight; returns samples in send order and wall time."""
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base, limits=limits, timeout=60,
                                 headers={"authorization": f"Bearer {key}"}) as client:
        async def one(body: dict) -> Sample:
            async with sem:
                t = time.perf_counter()
                r = await client.post("/v1/chat/completions", json=body)
                return Sample(r.status_code, time.perf_counter() - t, r.headers.get("x-gateway-provider"),
                              r.headers.get("x-gateway-cache"))
        started = time.perf_counter()
        samples = await asyncio.gather(*(one(b) for b in bodies))
        return list(samples), time.perf_counter() - started


def ms(seconds: list[float], q: float) -> float:
    if not seconds:
        return float("nan")
    if len(seconds) == 1:
        return round(seconds[0] * 1000, 1)
    return round(statistics.quantiles(seconds, n=100, method="inclusive")[int(q) - 1] * 1000, 1)


def ok_rate(samples: list[Sample]) -> float:
    return round(100 * sum(s.status == 200 for s in samples) / len(samples), 1)


def body(model: str, messages=QUESTION, **extra) -> dict:
    return {"model": model, "messages": messages, **extra}


# --------------------------------------------------------------------------- scenarios
async def overhead(n: int) -> dict:
    """Latency the gateway adds. With an instant provider, all of it is the gateway's: auth lookup,
    rate limiter, redaction, the failure policy, the ledger write and the log line. With a provider
    that takes 200 ms, what's left above 200 ms is the gateway's share."""
    results = {}
    modes = {"before": False, "after": True}       # after = cached key lookups + batched ledger writes
    for mode, fast in modes.items():
        for provider_ms in (0, 200):
            scenario = {"providers": [{"name": "model", "latency_s": provider_ms / 1000}],
                        "routes": {"m": [{"provider": "model", "model": "m-1"}]}, "fast_path": fast}
            async with gateway(scenario) as (base, key):
                await fire(base, key, [body("m")] * 50, 10)                       # warm up
                for concurrency in (1, 10, 50):
                    # one at a time against a 200 ms provider is slow and very steady: fewer requests do
                    count = max(20, n // 5) if concurrency == 1 and provider_ms else n
                    samples, wall = await fire(base, key, [body("m")] * count, concurrency)
                    lat = [s.seconds for s in samples]
                    results[f"{mode}:{provider_ms}ms@{concurrency}"] = {
                        "mode": mode, "provider_ms": provider_ms, "concurrency": concurrency, "requests": count,
                        "p50_ms": ms(lat, 50), "p95_ms": ms(lat, 95), "p99_ms": ms(lat, 99),
                        "added_p50_ms": round(ms(lat, 50) - provider_ms, 1), "added_p95_ms": round(ms(lat, 95) - provider_ms, 1),
                        "requests_per_s": round(count / wall, 1), "ok_pct": ok_rate(samples)}
    return results


async def flaky(n: int) -> dict:
    """A provider failing 30% of calls with 5xx errors, behind three levels of protection."""
    providers = [{"name": "flaky", "failure_rate": 0.3, "latency_s": 0.02, "seed": 7},
                 {"name": "backup", "latency_s": 0.02}]
    setups = {
        "no protection (1 try, no fallback)": ({"m": [{"provider": "flaky", "model": "f"}]},
                                               {"attempts_per_target": 1}),
        "retry only (2 tries)": ({"m": [{"provider": "flaky", "model": "f"}]}, {"attempts_per_target": 2}),
        "retry + fallback": ({"m": [{"provider": "flaky", "model": "f"}, {"provider": "backup", "model": "b"}]},
                             {"attempts_per_target": 2}),
    }
    results = {}
    for name, (routes, policy) in setups.items():
        policy |= {"backoff_s": 0.01}
        async with gateway({"providers": providers, "routes": routes, "policy": policy}) as (base, key):
            samples, _ = await fire(base, key, [body("m")] * n, 20)
        lat = [s.seconds for s in samples if s.status == 200]
        results[name] = {"requests": n, "ok_pct": ok_rate(samples), "p50_ms": ms(lat, 50), "p95_ms": ms(lat, 95),
                         "served_by_backup_pct": round(100 * sum(s.provider == "backup" for s in samples) / n, 1)}
    return results


async def outage(n: int) -> dict:
    """The first provider hangs (never answers); the gateway's per-call timeout is 1 s and the backup
    answers in 50 ms. Without a circuit breaker every request waits out the timeout first; with
    one, after 5 timeouts the hung provider is skipped."""
    providers = [{"name": "hung", "latency_s": 600}, {"name": "backup", "latency_s": 0.05}]
    routes = {"m": [{"provider": "hung", "model": "h"}, {"provider": "backup", "model": "b"}]}
    results = {}
    for name, threshold in (("without circuit breaker", 10**9), ("with circuit breaker", 5)):
        policy = {"attempt_timeout_s": 1.0, "breaker_threshold": threshold, "breaker_cooldown_s": 300}
        async with gateway({"providers": providers, "routes": routes, "policy": policy}) as (base, key):
            samples, wall = await fire(base, key, [body("m")] * n, 10)
        lat = [s.seconds for s in samples]
        results[name] = {"requests": n, "ok_pct": ok_rate(samples), "mean_ms": round(statistics.mean(lat) * 1000, 1),
                         "p50_ms": ms(lat, 50), "p95_ms": ms(lat, 95), "total_s": round(wall, 1),
                         "paid_the_timeout": sum(s.seconds >= 1.0 for s in samples)}
    return results


async def cache(n: int) -> dict:
    """Repeat questions: 100 distinct prompts, popularity skewed like real traffic (a few questions
    asked very often, a long tail asked rarely), all at temperature 0; the provider takes 300 ms."""
    rng = random.Random(7)
    weights = [1 / rank for rank in range(1, 101)]                  # Zipf-like popularity
    prompts = rng.choices(range(100), weights=weights, k=n)
    bodies = [body("m", [{"role": "user", "content": f"Question {p}: how do I reset my password?"}], temperature=0)
              for p in prompts]
    scenario = {"providers": [{"name": "model", "latency_s": 0.3}], "routes": {"m": [{"provider": "model", "model": "m-1"}]},
                "cache": True}
    async with gateway(scenario) as (base, key):
        samples, wall = await fire(base, key, bodies, 10)
    hits = [s.seconds for s in samples if s.cache == "hit"]
    misses = [s.seconds for s in samples if s.cache == "miss"]
    calls = len(misses)
    # Illustrative cost: gemini-2.5-flash list prices ($0.30 / $2.50 per 1M tokens) at an assumed
    # 500 input and 300 output tokens per request. The fake provider's own counts aren't real tokens.
    per_call = 500 * 0.30e-6 + 300 * 2.50e-6
    return {"requests": n, "distinct_prompts": len(set(prompts)), "hit_pct": round(100 * len(hits) / n, 1),
            "provider_calls": calls, "calls_saved": n - calls,
            "hit_p50_ms": ms(hits, 50), "miss_p50_ms": ms(misses, 50), "total_s": round(wall, 1),
            "est_cost_without_cache_usd": round(n * per_call, 4), "est_cost_with_cache_usd": round(calls * per_call, 4),
            "repeat_misses": calls - len(set(prompts))}


# --------------------------------------------------------------------------- report
def report(results: dict) -> str:
    m = results["machine"]
    o, f, out, c = results["overhead"], results["flaky"], results["outage"], results["cache"]
    lines = [
        "# Benchmark results",
        "",
        f"Measured {results['date']} on {m['cpus']} vCPU ({m['cpu']}), Python {m['python']}, ledger on {m['database']}, "
        "one uvicorn worker. The load generator runs on the same machine, so at concurrency 50 it competes with the "
        "gateway for CPU and those rows understate what the gateway alone can do. Reproduce with `python -m bench.run`; "
        "numbers vary by machine.",
        "",
        "## Overhead: latency the gateway adds",
        "",
        "Provider latency is fixed, so anything above it is the gateway: auth, rate limiter, redaction, "
        "failure policy, ledger and log line. *Before* looks up the API key and writes the ledger row in the "
        "database on every request; *after* caches valid keys for 30 s and writes ledger rows in background batches.",
        "",
        "| | Provider latency | Concurrency | p50 | p95 | Added at p50 | Added at p95 | Requests/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in o.values():
        lines.append(f"| {r['mode']} | {r['provider_ms']} ms | {r['concurrency']} | {r['p50_ms']} ms | {r['p95_ms']} ms | "
                     f"{r['added_p50_ms']} ms | {r['added_p95_ms']} ms | {r['requests_per_s']} |")
    lines += ["", "## A provider failing 30% of calls", "",
              "| Setup | Success | p50 | p95 | Served by backup |", "|---|---|---|---|---|"]
    for name, r in f.items():
        lines.append(f"| {name} | {r['ok_pct']}% | {r['p50_ms']} ms | {r['p95_ms']} ms | {r['served_by_backup_pct']}% |")
    lines += ["", "## A provider that hangs (1 s timeout, backup answers in 50 ms)", "",
              "| Setup | Success | Mean | p50 | p95 | Requests that waited out the timeout | Total time |",
              "|---|---|---|---|---|---|---|"]
    for name, r in out.items():
        lines.append(f"| {name} | {r['ok_pct']}% | {r['mean_ms']} ms | {r['p50_ms']} ms | {r['p95_ms']} ms | "
                     f"{r['paid_the_timeout']} of {r['requests']} | {r['total_s']} s |")
    lines += ["", "## Response cache on repeat questions", "",
              f"{c['requests']} temperature-0 requests over {c['distinct_prompts']} distinct prompts with skewed popularity; "
              "the provider takes 300 ms.", "",
              "| Hit rate | Provider calls | Hit p50 | Miss p50 | Est. cost without cache | Est. cost with cache |",
              "|---|---|---|---|---|---|",
              f"| {c['hit_pct']}% | {c['provider_calls']} of {c['requests']} | {c['hit_p50_ms']} ms | {c['miss_p50_ms']} ms | "
              f"${c['est_cost_without_cache_usd']} | ${c['est_cost_with_cache_usd']} |", "",
              "Costs are illustrative: gemini-2.5-flash list prices at an assumed 500 input and 300 output tokens per "
              f"request. {c['repeat_misses']} misses were repeats of a prompt already in flight: the cache doesn't merge "
              "identical concurrent requests, so the first few callers of a popular question each reach the provider.",
              ""]
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small runs, for a smoke test")
    args = parser.parse_args()
    scale = 0.2 if args.quick else 1.0
    n = lambda count: max(20, int(count * scale))  # noqa: E731

    results = {"date": time.strftime("%Y-%m-%d"),
               "machine": {"cpus": os.cpu_count(), "cpu": _cpu_name(), "python": platform.python_version(),
                           "database": "Postgres" if os.environ.get("BENCH_DATABASE_URL") else "SQLite"}}
    for name, fn, count in (("overhead", overhead, 1000), ("flaky", flaky, 1000), ("outage", outage, 200),
                            ("cache", cache, 1000)):
        print(f"running {name}...", flush=True)
        results[name] = await fn(n(count))
    out = Path(__file__).parent
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "RESULTS.md").write_text(report(results))
    print(report(results))


def _cpu_name() -> str:
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    with contextlib.suppress(Exception):
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              check=True).stdout.strip()
    return platform.processor() or "unknown CPU"


if __name__ == "__main__":
    asyncio.run(main())
