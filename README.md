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
| 3 | Timeouts, retries, fallback across a route's targets, circuit breaker | |
| 4 | Rate limits, response cache | |
| 5 | Streaming, PII redaction | |
| 6 | Load, chaos and cache benchmarks | |
| 7 | Deploy, demo, "why not LiteLLM" | |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add provider settings; with none, only the fake provider is on
alembic upgrade head          # create the tables (SQLite by default, Postgres via DATABASE_URL)
python -m gateway.keys create --name my-app --budget 5    # prints the key once
uvicorn gateway.main:app --reload
```

Keys: `python -m gateway.keys list` and `python -m gateway.keys revoke <prefix>`. A key's spend this month:

```bash
curl -H "Authorization: Bearer gw_..." localhost:8000/v1/usage
```

`routes.yaml` maps the model names clients send to provider models. The first target serves the request. From Day 3, the later targets are the fallback order:

```yaml
routes:
  fast:
    - provider: gemini
      model: gemini-2.5-flash
    - provider: bedrock
      model: us.amazon.nova-2-lite-v1:0
```

The gateway refuses to start if a route names a provider that isn't configured. A typo shows up at boot, not as an error on the first request.

## Design choices

- **The provider-neutral request is the only thing adapters see.** The wire format goes through `normalize()` once. A new provider never has to deal with OpenAI quirks such as `developer` messages, `max_completion_tokens` or a `stop` given as a string.
- **Unsupported options are refused with a 400, not ignored.** `n>1`, `json_schema`, tool calls, images and (until Day 5) streaming fail loudly. A client that silently got one choice instead of three would be a worse bug.
- **Token counts are never invented.** If a provider doesn't report usage, `usage` is `null`. Costs built on usage are labeled as estimates.
- **Errors keep OpenAI's shape, and the status shows whose fault it was.** 4xx means the request was wrong. 502 or 504 means a provider failed. Every provider error records whether a retry could help, which is what fallback (Day 3) uses. Provider error text never echoes the gateway's credentials.
- **API keys are stored only as SHA-256 hashes.** A leaked database doesn't leak usable keys, and a key is shown once, when it's created. Plain SHA-256 is enough because keys are 256-bit random values; password hashing (bcrypt and similar) exists for low-entropy secrets.
- **Every accepted request is a ledger row,** successes and provider failures alike, with the provider's own token counts and an estimated cost. Requests rejected before reaching a provider (bad key, unknown model, invalid options) are not recorded. Money is stored as integer micro-dollars, so sums are exact.
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
