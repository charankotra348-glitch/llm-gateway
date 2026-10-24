# LLM Gateway

One OpenAI-compatible endpoint in front of several model providers (Gemini, Amazon Bedrock, Ollama). An application sends ordinary OpenAI requests. The gateway decides which provider serves each one and turns provider failures into errors the client can act on.

Any existing OpenAI client works without code changes. Only the base URL changes:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="gw_...")   # a gateway key
reply = client.chat.completions.create(model="demo", messages=[{"role": "user", "content": "hi"}])
```

## Status

| Day | Scope | State |
|---|---|---|
| 1 | OpenAI-compatible `/v1/chat/completions` and `/v1/models`, adapters for Gemini, Bedrock, Ollama and a fake provider, OpenAI-format errors | done |
| 2 | API keys, Postgres request ledger with cost estimates, per-key monthly budgets, `/v1/usage` | done |
| 3 | Timeouts, retries, fallback across a route's targets, circuit breakers, `/v1/status` | done |
| 4 | Per-key rate limits, response cache | done |
| 5 | Streaming, PII redaction | |
| 6 | Load, chaos and cache benchmarks | |
| 7 | Deploy, demo, "why not LiteLLM" | |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add provider settings; with none, only the fake provider is on
alembic upgrade head          # create the tables (SQLite by default, Postgres via DATABASE_URL)
python -m gateway.keys create --name my-app --budget 5 --rpm 30   # prints the key once
uvicorn gateway.main:app --reload
```

Keys: `python -m gateway.keys list` and `python -m gateway.keys revoke <prefix>`. A key's spend this month, and the circuit breaker state of each provider:

```bash
curl -H "Authorization: Bearer gw_..." localhost:8000/v1/usage
curl -H "Authorization: Bearer gw_..." localhost:8000/v1/status
```

The shipped `resilient` route shows fallback with no API keys. Its first target, `flaky`, fails half its calls, and `fake` catches them. Send it 20 requests: all succeed, `x-gateway-provider` shows who answered, and `/v1/status` shows `flaky`'s circuit open.

`routes.yaml` maps the model names clients send to provider models. Targets are tried in order until one answers:

```yaml
routes:
  fast:
    - provider: gemini
      model: gemini-2.5-flash
    - provider: bedrock
      model: us.amazon.nova-2-lite-v1:0
```

The gateway refuses to start if a route names a provider that isn't configured. A typo shows up at boot, not as an error on the first request.

## When a provider fails

| What went wrong | What the gateway does | Why |
|---|---|---|
| Provider down (5xx, connection refused) | Retry the same target with jittered backoff (2 tries by default), then the next target | Often a blip; a second try is cheap |
| Timeout | Next target immediately | A retry would wait just as long again |
| Provider rate limit (429) | Next target immediately | A retry would be throttled again |
| Gateway's provider credentials rejected | Next target immediately | This target is misconfigured; others may be fine |
| Provider rejects the request (bad input) | Stop and return the error | Another provider would most likely reject it too |

Each target (provider + model) has a **circuit breaker**. After 5 consecutive failures (`BREAKER_THRESHOLD`) its circuit opens, and requests skip it without paying a timeout to rediscover the outage. After 30 s (`BREAKER_COOLDOWN_S`) one request is let through as a probe: if it succeeds the circuit closes, if not it stays open for another cooldown.

Two clocks bound every request. Each provider call gets `REQUEST_TIMEOUT_S`, enforced on the wall clock as well as by the HTTP client: an HTTP read timeout is per chunk, so a provider trickling bytes could otherwise run past it. The whole request, across retries and fallbacks, gets `TOTAL_TIMEOUT_S`.

The client sees one normal response. `x-gateway-provider` says who answered and `x-gateway-attempts` how many provider calls it took; the ledger records the same. If every target fails, the error says what each one returned, with the status of the last.

Limits worth knowing: breaker state lives in each server process, so replicas learn about an outage separately (shared state, such as Redis, would fix that), and a retry after a timeout can't know whether the provider is still generating, and billing, the first answer.

## Rate limits and caching

**Rate limits.** Each key has a requests-per-minute limit (`--rpm` when creating it, otherwise `DEFAULT_RPM`, 60; `0` means none), enforced with a token bucket: a burst of up to `rpm` requests, then one per `60/rpm` seconds. Over the limit is a 429 `rate_limit_exceeded` with `retry-after`, which the OpenAI SDK reads, so a client with retries on waits and carries on without seeing an error. Successful responses carry `x-ratelimit-limit-requests` and `x-ratelimit-remaining-requests`, as OpenAI's do.

**Response cache.** An identical request (same key, route, messages and options) is answered from memory: no provider call, no cost, `x-gateway-cache: hit`, and a ledger row with provider `cache` and cost 0. `/v1/usage` counts the hits.

- Only `temperature: 0` requests are cached. Any other temperature asks for a sample, and replaying one sample would quietly change what the request means; those get `x-gateway-cache: bypass`.
- Entries belong to one API key. Another key never receives the answer, nor learns from a fast response that someone else asked the same thing.
- Standard `Cache-Control` works: `no-cache` skips the lookup but stores the fresh answer, `no-store` does neither.
- Failures are never cached. Entries live for `CACHE_TTL_S` (1 hour), and the least recently used go first past `CACHE_MAX_ENTRIES`.
- Hits are free, so they're served even when a key's budget is spent; rate limits still count them.

Both keep their state in the server process, like the circuit breakers: with several replicas each has its own buckets and cache, so the effective limit is `rpm × replicas`. Redis would make them shared.

## Design choices

- **The provider-neutral request is the only thing adapters see.** The wire format goes through `normalize()` once. A new provider never has to deal with OpenAI quirks such as `developer` messages, `max_completion_tokens` or a `stop` given as a string.
- **Unsupported options are refused with a 400, not ignored.** `n>1`, `json_schema`, tool calls, images and (until Day 5) streaming fail loudly. A client that silently got one choice instead of three would be a worse bug.
- **Token counts are never invented.** If a provider doesn't report usage, `usage` is `null`. Costs built on usage are labeled as estimates.
- **Errors keep OpenAI's shape, and the status shows whose fault it was.** 4xx means the request was wrong. 502 or 504 means a provider failed. Every provider error records whether a retry could help, which the fallback policy uses. Provider error text never echoes the gateway's credentials.
- **API keys are stored only as SHA-256 hashes.** A leaked database doesn't leak usable keys, and a key is shown once, when it's created. Plain SHA-256 is enough because keys are 256-bit random values; password hashing (bcrypt and similar) exists for low-entropy secrets.
- **Every request that reaches a provider is a ledger row,** successes and failures alike, with who answered, how many calls it took, the provider's own token counts and an estimated cost. Requests turned away before any provider call (bad key, unknown model, invalid options, every circuit open) are not recorded. Money is stored as integer micro-dollars, so sums are exact.
- **Budgets are monthly (UTC) and soft.** The check runs before each request against the month's recorded spend. Requests already in flight can take a key slightly past its limit, bounded by concurrency times the cost of one request. A hard limit would need a reservation per request; that's a deliberate trade-off.
- **Budget errors are 429s the client won't retry.** The OpenAI SDKs retry every 429 by default, so every error carries `x-should-retry`: `false` for budgets and bad requests, `true` for provider timeouts, rate limits and outages.
- **Unpriced requests are visible, not hidden.** Requests with no reported usage, or to a model missing from `prices.yaml`, get no cost (not $0). They don't count toward the budget, `/v1/usage` reports how many there were, and the gateway warns at startup about any route target it can't price.
- **A ledger write that fails is logged and doesn't fail the response.** The provider has already charged for the answer, so throwing it away would cost twice.
- **Bedrock credentials come from the AWS chain** (an IAM role on AWS, `aws configure` locally), never from gateway settings.
- **The fake provider is in-process.** It has configurable latency and failure rate and a seed, so tests, demos and the Day 6 chaos tests need no API keys or network.

## Tests

```bash
ruff check . && pytest -q
```

`tests/test_openai_sdk.py` drives the gateway with the official `openai` package: success, finish reasons, `/v1/models`, and the SDK exception each error status maps to. The provider tests mock HTTP with `httpx.MockTransport` and Bedrock with botocore's `Stubber`, so no test touches the network.

The test database is built with the real migrations. Tests use SQLite by default. Set `TEST_DATABASE_URL=postgresql+asyncpg://...` to run them on Postgres. CI does both, and also checks that the migrations match the models (`alembic check`).
