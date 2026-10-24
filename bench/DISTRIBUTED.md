# Distributed benchmark

Measured 2026-10-24 on 8 vCPU (Apple M1), Redis 8.10.2, ledger on SQLite. 3 gateway processes on one machine; requests are spread round-robin, as a load balancer would. Each scenario runs with every process keeping its own state, then with the state shared through Redis. Reproduce with `BENCH_REDIS_URL=redis://localhost:6379/15 python -m bench.distributed`.

## One rate limit, three processes

A key limited to 30 requests a minute sends 90 at once.

| State | Allowed | Refused (429) |
|---|---|---|
| each process alone | 90 of 90 | 0 |
| shared through Redis | 30 of 90 | 60 |

## Identical requests in flight

60 identical temperature-0 requests arrive at once, 20 at each process; the provider takes 500 ms.

| Setup | Provider calls | Answered by another request in flight | p50 |
|---|---|---|---|
| cache only, each process alone | 60 | 0 | 549.9 ms |
| cache + coalescing, each process alone | 3 | 57 | 557.6 ms |
| cache + coalescing, shared through Redis | 1 | 59 | 571.4 ms |

## A provider that hangs

1 s timeout, backup answers in 50 ms, 60 requests with one in flight per process; breakers open after 5 timeouts.

| State | Success | Requests that waited out the timeout | Mean | Total time |
|---|---|---|---|---|
| each process alone | 60 of 60 | 15 | 310.9 ms | 6.2 s |
| shared through Redis | 60 of 60 | 6 | 160.5 ms | 3.2 s |

## What Redis costs

An instant provider and requests the cache can't answer: each request makes a rate-limit call and two circuit-breaker calls to Redis.

| State | p50, one process, one request at a time | p50, three processes, 30 in flight | Requests/s, three processes |
|---|---|---|---|
| each process alone | 0.8 ms | 23.6 ms | 1033.1 |
| shared through Redis | 1.1 ms | 25.2 ms | 908.6 |
