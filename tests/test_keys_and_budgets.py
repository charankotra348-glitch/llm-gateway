from datetime import UTC, datetime, timedelta

import openai
import pytest
from sqlalchemy import select

from gateway.errors import ProviderUnavailable
from gateway.keys import hash_key
from gateway.ledger import month_start
from gateway.models import ApiKey, RequestLog
from gateway.pricing import PriceTable

MSG = [{"role": "user", "content": "hello there"}]


def sdk_for(client, key):
    return openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=client, max_retries=0)


def ledger(db):
    async def rows(s):
        return list(await s.scalars(select(RequestLog).order_by(RequestLog.created_at)))
    return db(rows)


# --------------------------------------------------------------------------- auth
@pytest.mark.parametrize("headers", [{}, {"authorization": "Bearer"}, {"authorization": "Basic abc"},
                                     {"authorization": "Bearer gw_not-a-real-key"},
                                     {"authorization": "Bearer sk-openai-style"}])
def test_requests_without_a_valid_key_are_401(client, headers):
    r = client.post("/v1/chat/completions", json={"model": "demo", "messages": MSG}, headers=headers)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_sdk_raises_authentication_error(client):
    with pytest.raises(openai.AuthenticationError):
        sdk_for(client, "gw_wrong").chat.completions.create(model="demo", messages=MSG)


def test_models_needs_a_key_too(client, api_key):
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"authorization": f"Bearer {api_key}"}).status_code == 200


def test_revoked_key_stops_working(client, db, make_key):
    row, key = make_key()
    sdk_for(client, key).chat.completions.create(model="demo", messages=MSG)

    async def revoke(s):
        (await s.get(ApiKey, row.id)).revoked_at = datetime.now(UTC)
        await s.commit()
    db(revoke)
    with pytest.raises(openai.AuthenticationError):
        sdk_for(client, key).chat.completions.create(model="demo", messages=MSG)


def test_only_the_hash_is_stored(db, make_key):
    row, key = make_key()
    stored = db(lambda s: s.get(ApiKey, row.id))
    assert stored.key_hash == hash_key(key) and key not in (stored.key_hash, stored.prefix)
    assert key.startswith(stored.prefix) and len(key) > 40


# --------------------------------------------------------------------------- ledger
def test_every_request_is_in_the_ledger_with_cost(client, db, make_key):
    row, key = make_key()
    resp = sdk_for(client, key).chat.completions.create(model="demo", messages=MSG)
    [entry] = ledger(db)
    assert entry.id == resp.id and entry.key_id == row.id
    assert (entry.route, entry.provider, entry.model, entry.outcome) == ("demo", "fake", "fake-echo", "ok")
    assert (entry.input_tokens, entry.output_tokens) == (resp.usage.prompt_tokens, resp.usage.completion_tokens)
    # test prices: $1 per input token, $2 per output token
    assert entry.cost_micros == (entry.input_tokens * 1 + entry.output_tokens * 2) * 1_000_000


def test_failed_requests_are_recorded_at_no_cost(client, db, fake, api_key):
    async def boom(*a, **k):
        raise ProviderUnavailable("down")
    fake.complete = boom
    with pytest.raises(openai.APIStatusError):
        sdk_for(client, api_key).chat.completions.create(model="demo", messages=MSG)
    [entry] = ledger(db)
    assert entry.outcome == "provider_unavailable" and entry.cost_micros == 0 and entry.input_tokens is None


def test_bad_requests_never_reach_the_ledger(client, db, api_key):
    with pytest.raises(openai.NotFoundError):
        sdk_for(client, api_key).chat.completions.create(model="nope", messages=MSG)
    with pytest.raises(openai.BadRequestError):
        sdk_for(client, api_key).chat.completions.create(model="demo", messages=MSG, n=3)
    assert ledger(db) == []


def test_unknown_usage_is_unpriced_not_zero(client, db, fake, api_key):
    from gateway.schemas import Completion

    async def no_usage(request, *, model, timeout_s):
        return Completion(text="hi", model=model)
    fake.complete = no_usage
    sdk_for(client, api_key).chat.completions.create(model="demo", messages=MSG)
    assert ledger(db)[0].cost_micros is None
    usage = client.get("/v1/usage", headers={"authorization": f"Bearer {api_key}"}).json()
    assert usage["unpriced_requests"] == 1 and usage["estimated_spend_usd"] == 0


def test_ledger_failure_does_not_fail_the_response(client, gateway, api_key, monkeypatch):
    real = gateway.sessions
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 2:          # 1 = auth lookup, 2 = ledger write (no budget on this key)
            raise ConnectionError("database went away")
        return real()
    monkeypatch.setattr(gateway, "sessions", flaky)
    resp = sdk_for(client, api_key).chat.completions.create(model="demo", messages=MSG)
    assert resp.choices[0].message.content == "echo: hello there"


# --------------------------------------------------------------------------- budgets and usage
def test_budget_blocks_once_spent_and_is_not_retried(client, make_key):
    _, key = make_key(budget_usd=5)            # one request here costs well over $5
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=client, max_retries=3)
    sdk.chat.completions.create(model="demo", messages=MSG)
    with pytest.raises(openai.RateLimitError) as exc:
        sdk.chat.completions.create(model="demo", messages=MSG)
    assert exc.value.code == "budget_exceeded"
    assert exc.value.response.headers["x-should-retry"] == "false"


def test_budget_is_per_key(client, make_key):
    _, spent = make_key(budget_usd=5)
    _, other = make_key(budget_usd=5)
    sdk_for(client, spent).chat.completions.create(model="demo", messages=MSG)
    with pytest.raises(openai.RateLimitError):
        sdk_for(client, spent).chat.completions.create(model="demo", messages=MSG)
    sdk_for(client, other).chat.completions.create(model="demo", messages=MSG)


def test_last_months_spend_does_not_count(client, db, make_key):
    row, key = make_key(budget_usd=5)

    async def old_spend(s):
        s.add(RequestLog(id="chatcmpl-old", key_id=row.id, route="demo", provider="fake", model="fake-echo",
                         outcome="ok", input_tokens=1, output_tokens=1, cost_micros=50_000_000, latency_ms=1,
                         created_at=month_start() - timedelta(seconds=1)))
        await s.commit()
    db(old_spend)
    sdk_for(client, key).chat.completions.create(model="demo", messages=MSG)   # not blocked


def test_usage_endpoint(client, make_key):
    _, key = make_key(name="my-app", budget_usd=100)
    sdk_for(client, key).chat.completions.create(model="demo", messages=MSG)
    sdk_for(client, key).chat.completions.create(model="demo", messages=MSG)
    u = client.get("/v1/usage", headers={"authorization": f"Bearer {key}"}).json()
    assert u["name"] == "my-app" and u["requests"] == 2 and u["monthly_budget_usd"] == 100
    assert u["estimated_spend_usd"] == 2 * (2 * 1 + 3 * 2)     # "hello there" = 2 in, "echo: hello there" = 3 out


def test_retryable_errors_say_so(client, fake, api_key):
    async def boom(*a, **k):
        raise ProviderUnavailable("down")
    fake.complete = boom
    r = client.post("/v1/chat/completions", json={"model": "demo", "messages": MSG},
                    headers={"authorization": f"Bearer {api_key}"})
    assert r.status_code == 502 and r.headers["x-should-retry"] == "true"


# --------------------------------------------------------------------------- pricing
def test_pricing():
    table = PriceTable.from_dict({"free_providers": ["ollama"],
                                  "prices": {"gemini": {"gemini-2.5-flash": {"input_per_mtok": 0.30,
                                                                             "output_per_mtok": 2.50}}}})
    assert table.cost_micros("gemini", "gemini-2.5-flash", 1_000_000, 0) == 300_000        # $0.30
    assert table.cost_micros("gemini", "gemini-2.5-flash", 1000, 1000) == 2800             # $0.0028
    assert table.cost_micros("gemini", "gemini-2.5-pro", 10, 10) is None
    assert table.cost_micros("gemini", "gemini-2.5-flash", None, 10) is None
    assert table.cost_micros("ollama", "llama3.2", 500, 500) == 0


def test_shipped_price_file_covers_the_example_routes():
    table = PriceTable.load("prices.yaml")
    assert table.knows("gemini", "gemini-2.5-flash") and table.knows("bedrock", "us.amazon.nova-2-lite-v1:0")
    assert table.knows("fake", "anything") and table.knows("ollama", "anything")
