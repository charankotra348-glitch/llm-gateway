# Benchmark results

Measured 2026-10-24 on 2 vCPU (Intel(R) Xeon(R) Processor @ 2.10GHz), Python 3.13.16, ledger on SQLite, one uvicorn worker. The load generator runs on the same machine, so at concurrency 50 it competes with the gateway for CPU and those rows understate what the gateway alone can do. Reproduce with `python -m bench.run`; numbers vary by machine.

## Overhead: latency the gateway adds

Provider latency is fixed, so anything above it is the gateway: auth, rate limiter, redaction, failure policy, ledger and log line. *Before* looks up the API key and writes the ledger row in the database on every request; *after* caches valid keys for 30 s and writes ledger rows in background batches.

| | Provider latency | Concurrency | p50 | p95 | Added at p50 | Added at p95 | Requests/s |
|---|---|---|---|---|---|---|---|
| before | 0 ms | 1 | 6.0 ms | 7.8 ms | 6.0 ms | 7.8 ms | 161.5 |
| before | 0 ms | 10 | 21.0 ms | 148.9 ms | 21.0 ms | 148.9 ms | 192.2 |
| before | 0 ms | 50 | 217.4 ms | 651.3 ms | 217.4 ms | 651.3 ms | 181.1 |
| before | 200 ms | 1 | 209.6 ms | 211.7 ms | 9.6 ms | 11.7 ms | 4.8 |
| before | 200 ms | 10 | 210.4 ms | 219.3 ms | 10.4 ms | 19.3 ms | 46.9 |
| before | 200 ms | 50 | 227.4 ms | 452.0 ms | 27.4 ms | 252.0 ms | 175.8 |
| after | 0 ms | 1 | 1.8 ms | 2.5 ms | 1.8 ms | 2.5 ms | 536.0 |
| after | 0 ms | 10 | 10.3 ms | 26.1 ms | 10.3 ms | 26.1 ms | 784.3 |
| after | 0 ms | 50 | 147.7 ms | 726.8 ms | 147.7 ms | 726.8 ms | 208.3 |
| after | 200 ms | 1 | 203.3 ms | 204.4 ms | 3.3 ms | 4.4 ms | 4.9 |
| after | 200 ms | 10 | 203.9 ms | 207.6 ms | 3.9 ms | 7.6 ms | 48.8 |
| after | 200 ms | 50 | 203.0 ms | 269.2 ms | 3.0 ms | 69.2 ms | 229.1 |

## A provider failing 30% of calls

| Setup | Success | p50 | p95 | Served by backup |
|---|---|---|---|---|
| no protection (1 try, no fallback) | 66.4% | 25.8 ms | 88.4 ms | 0.0% |
| retry only (2 tries) | 88.7% | 25.6 ms | 56.5 ms | 0.0% |
| retry + fallback | 100.0% | 26.8 ms | 74.1 ms | 11.0% |

## A provider that hangs (1 s timeout, backup answers in 50 ms)

| Setup | Success | Mean | p50 | p95 | Requests that waited out the timeout | Total time |
|---|---|---|---|---|---|---|
| without circuit breaker | 100.0% | 1056.4 ms | 1054.0 ms | 1061.1 ms | 200 of 200 | 21.2 s |
| with circuit breaker | 100.0% | 105.7 ms | 53.0 ms | 110.2 ms | 10 of 200 | 2.1 s |

## Response cache on repeat questions

1000 temperature-0 requests over 98 distinct prompts with skewed popularity; the provider takes 300 ms.

| Hit rate | Provider calls | Hit p50 | Miss p50 | Est. cost without cache | Est. cost with cache |
|---|---|---|---|---|---|
| 87.5% | 125 of 1000 | 2.7 ms | 304.1 ms | $0.9 | $0.1125 |

Costs are illustrative: gemini-2.5-flash list prices at an assumed 500 input and 300 output tokens per request. 27 misses were repeats of a prompt already in flight: the cache doesn't merge identical concurrent requests, so the first few callers of a popular question each reach the provider.
