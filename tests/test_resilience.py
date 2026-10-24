"""Retries, fallback and circuit breakers: unit tests on the policy, then through the HTTP API."""

import time

import openai
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from gateway.db import make_engine, make_sessions
from gateway.errors import (
    ProviderAuthError,
    ProviderRateLimited,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from gateway.main import Gateway, create_app
from gateway.models import RequestLog
from gateway.pricing import PriceTable
from gateway.providers.fake import FakeProvider
from gateway.resilience import CircuitBreaker, Resilience, ResiliencePolicy
from gateway.routes import RouteTable
from gateway.schemas import ChatRequest, Message

pytestmark = pytest.mark.anyio
CHAT = ChatRequest(messages=[Message(role="user", text="hi")])
MSG = [{"role": "user", "content": "hi"}]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def two_targets():
    return RouteTable.from_dict({"routes": {"smart": [{"provider": "a", "model": "a-1"},
                                                      {"provider": "b", "model": "b-1"}]}}).get("smart")


def resilience(**kw):
    kw.setdefault("backoff_s", 0)
    return Resilience(ResiliencePolicy(**kw))


# --------------------------------------------------------------------------- circuit breaker
def test_breaker_opens_after_threshold_and_probes_after_cooldown():
    clock = Clock()
    b = CircuitBreaker(threshold=3, cooldown_s=30, clock=clock)
    for _ in range(2):
        b.failure()
    assert b.state == "closed" and b.allow()
    b.failure()
    assert b.state == "open" and not b.allow()

    clock.now += 29.9
    assert not b.allow()
    clock.now += 0.1
    assert b.state == "half_open"
    assert b.allow()              # the one probe
    assert not b.allow()          # everyone else waits for it
    b.success()
    assert b.state == "closed" and b.failures == 0


def test_failed_probe_reopens_for_a_full_cooldown():
    clock = Clock()
    b = CircuitBreaker(threshold=1, cooldown_s=10, clock=clock)
    b.failure()
    clock.now += 10
    assert b.allow()
    b.failure()
    assert b.state == "open"
    clock.now += 9
    assert not b.allow()


def test_a_success_resets_the_count():
    b = CircuitBreaker(threshold=3)
    b.failure(), b.failure(), b.success(), b.failure(), b.failure()
    assert b.state == "closed"


def test_released_probe_lets_the_next_request_probe():
    clock = Clock()
    b = CircuitBreaker(threshold=1, cooldown_s=1, clock=clock)
    b.failure()
    clock.now += 1
    assert b.allow()
    b.release()                   # the probe was a bad request: says nothing about health
    assert b.state == "half_open" and b.allow()


# --------------------------------------------------------------------------- the policy
async def test_first_target_answers():
    a, b = FakeProvider(), FakeProvider()
    result = await resilience().complete(two_targets(), CHAT, {"a": a, "b": b})
    assert result.target.provider == "a" and [x.outcome for x in result.attempts] == ["ok"]
    assert b.calls == []


async def test_unavailable_is_retried_on_the_same_target():
    a = FakeProvider(script=[ProviderUnavailable("blip")])
    result = await resilience().complete(two_targets(), CHAT, {"a": a, "b": FakeProvider()})
    assert result.target.provider == "a"
    assert [x.outcome for x in result.attempts] == ["provider_unavailable", "ok"]


async def test_retry_waits_with_backoff():
    waits = []

    async def sleep(s):
        waits.append(s)
    r = Resilience(ResiliencePolicy(backoff_s=0.2, attempts_per_target=3), sleep=sleep)
    a = FakeProvider(script=[ProviderUnavailable("1"), ProviderUnavailable("2")])
    await r.complete(two_targets(), CHAT, {"a": a, "b": FakeProvider()})
    assert len(waits) == 2 and 0 <= waits[0] <= 0.2 and 0 <= waits[1] <= 0.4


async def test_target_that_stays_down_falls_back():
    a = FakeProvider(script=[ProviderUnavailable("down")] * 2)
    b = FakeProvider()
    result = await resilience().complete(two_targets(), CHAT, {"a": a, "b": b})
    assert result.target.provider == "b"
    assert [(x.provider, x.outcome) for x in result.attempts] == [
        ("a", "provider_unavailable"), ("a", "provider_unavailable"), ("b", "ok")]


@pytest.mark.parametrize("error", [ProviderTimeout("slow"), ProviderRateLimited("429"), ProviderAuthError("key")])
async def test_these_errors_skip_straight_to_the_next_target(error):
    a = FakeProvider(script=[error])
    result = await resilience().complete(two_targets(), CHAT, {"a": a, "b": FakeProvider()})
    assert len(a.calls) == 1 and result.target.provider == "b"


async def test_a_rejected_request_is_not_sent_anywhere_else():
    a, b = FakeProvider(script=[ProviderRejected("bad stop sequence")]), FakeProvider()
    with pytest.raises(ProviderRejected) as exc:
        await resilience().complete(two_targets(), CHAT, {"a": a, "b": b})
    assert b.calls == [] and [x.outcome for x in exc.value.attempts] == ["provider_rejected_request"]


async def test_all_targets_failing_raises_the_last_error_with_the_trail():
    a = FakeProvider(script=[ProviderTimeout("slow")])
    b = FakeProvider(script=[ProviderUnavailable("down")] * 2)
    with pytest.raises(ProviderUnavailable) as exc:
        await resilience().complete(two_targets(), CHAT, {"a": a, "b": b})
    assert "a/a-1: provider_timeout" in exc.value.message and "b/b-1: provider_unavailable" in exc.value.message
    assert len(exc.value.attempts) == 3


async def test_open_circuit_skips_the_target_without_calling_it():
    r = resilience(breaker_threshold=2)
    a, b = FakeProvider(script=[ProviderTimeout("slow")] * 2), FakeProvider()
    await r.complete(two_targets(), CHAT, {"a": a, "b": b})
    await r.complete(two_targets(), CHAT, {"a": a, "b": b})
    assert len(a.calls) == 2
    result = await r.complete(two_targets(), CHAT, {"a": a, "b": b})
    assert len(a.calls) == 2                       # not called: circuit open
    assert [x.outcome for x in result.attempts] == ["circuit_open", "ok"]
    assert {s["provider"]: s["state"] for s in r.states()} == {"a": "open", "b": "closed"}


async def test_circuit_closes_again_after_a_good_probe():
    clock = Clock()
    r = Resilience(ResiliencePolicy(breaker_threshold=1, breaker_cooldown_s=30, backoff_s=0), clock=clock)
    a, b = FakeProvider(script=[ProviderTimeout("slow")]), FakeProvider()
    await r.complete(two_targets(), CHAT, {"a": a, "b": b})
    clock.now += 30
    result = await r.complete(two_targets(), CHAT, {"a": a, "b": b})
    assert result.target.provider == "a" and r.breaker(two_targets().targets[0]).state == "closed"


async def test_every_circuit_open_is_a_retryable_unavailable():
    r = resilience(breaker_threshold=1)
    a, b = FakeProvider(script=[ProviderTimeout("x")]), FakeProvider(script=[ProviderTimeout("x")])
    with pytest.raises(ProviderTimeout):
        await r.complete(two_targets(), CHAT, {"a": a, "b": b})
    with pytest.raises(ProviderUnavailable) as exc:
        await r.complete(two_targets(), CHAT, {"a": a, "b": b})
    assert exc.value.retryable and "circuit open" in exc.value.message
    assert [x.outcome for x in exc.value.attempts] == ["circuit_open", "circuit_open"]


async def test_wall_clock_timeout_catches_a_provider_that_ignores_its_own():
    class Stuck(FakeProvider):
        async def complete(self, request, *, model, timeout_s):
            import anyio
            await anyio.sleep(10)          # ignores timeout_s entirely

    r = resilience(attempt_timeout_s=0.05)
    started = time.perf_counter()
    result = await r.complete(two_targets(), CHAT, {"a": Stuck(), "b": FakeProvider()})
    assert time.perf_counter() - started < 1
    assert [x.outcome for x in result.attempts] == ["provider_timeout", "ok"]


async def test_total_deadline_caps_the_whole_request():
    slow = FakeProvider(latency_s=5)
    r = resilience(attempt_timeout_s=10, total_timeout_s=0.1)
    started = time.perf_counter()
    with pytest.raises(ProviderTimeout):
        await r.complete(two_targets(), CHAT, {"a": slow, "b": FakeProvider(latency_s=5)})
    assert time.perf_counter() - started < 1


# --------------------------------------------------------------------------- through the API
@pytest.fixture
def api(database_url, make_key):
    """An app whose `smart` route is a -> b, both fakes."""
    a, b = FakeProvider(), FakeProvider()
    routes = RouteTable.from_dict({"routes": {"smart": [{"provider": "a", "model": "a-1"},
                                                        {"provider": "b", "model": "b-1"}]}})
    engine = make_engine(database_url)
    gw = Gateway({"a": a, "b": b}, routes, make_sessions(engine),
                 prices=PriceTable.from_dict({"free_providers": ["a", "b"]}),
                 resilience=resilience(breaker_threshold=2), engine=engine)
    _, key = make_key()
    with TestClient(create_app(gw)) as client:
        sdk = openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=client, max_retries=0)
        yield sdk, client, key, a, b


def ledger(db):
    async def rows(s):
        return list(await s.scalars(select(RequestLog)))
    return db(rows)


def test_fallback_is_invisible_to_the_client_except_in_headers(api, db):
    sdk, _, _, a, b = api
    a.script = [ProviderTimeout("slow")]
    raw = sdk.chat.completions.with_raw_response.create(model="smart", messages=MSG)
    assert raw.parse().choices[0].message.content == "echo: hi"
    assert raw.headers["x-gateway-provider"] == "b" and raw.headers["x-gateway-attempts"] == "2"
    [row] = ledger(db)
    assert (row.provider, row.model, row.attempts, row.outcome) == ("b", "b-1", 2, "ok")


def test_total_failure_is_recorded_against_the_last_target(api, db):
    sdk, _, _, a, b = api
    a.script = [ProviderTimeout("slow")]
    b.script = [ProviderRateLimited("busy")]
    with pytest.raises(openai.RateLimitError) as exc:
        sdk.chat.completions.create(model="smart", messages=MSG)
    assert exc.value.response.headers["x-should-retry"] == "true"
    [row] = ledger(db)
    assert (row.provider, row.attempts, row.outcome, row.cost_micros) == ("b", 2, "provider_rate_limited", 0)


def test_status_shows_breaker_states(api):
    sdk, client, key, a, b = api
    a.script = [ProviderTimeout("x")] * 2
    for _ in range(2):
        sdk.chat.completions.create(model="smart", messages=MSG)
    states = client.get("/v1/status", headers={"authorization": f"Bearer {key}"}).json()["targets"]
    assert {(s["provider"], s["state"]) for s in states} == {("a", "open"), ("b", "closed")}
    assert client.get("/v1/status").status_code == 401


def test_requests_turned_away_by_open_circuits_are_not_billed_or_recorded(api, db):
    sdk, _, _, a, b = api
    a.script = [ProviderTimeout("x")] * 2
    b.script = [ProviderTimeout("x")] * 2
    for _ in range(2):
        with pytest.raises(openai.APIStatusError):
            sdk.chat.completions.create(model="smart", messages=MSG)
    with pytest.raises(openai.APIStatusError) as exc:
        sdk.chat.completions.create(model="smart", messages=MSG)
    assert exc.value.status_code == 502 and "circuit open" in exc.value.message
    assert len(ledger(db)) == 2


def test_attempts_header_counts_calls_not_skipped_circuits(api, db):
    sdk, _, _, a, b = api
    a.script = [ProviderTimeout("x")] * 2
    for _ in range(2):
        sdk.chat.completions.create(model="smart", messages=MSG)
    raw = sdk.chat.completions.with_raw_response.create(model="smart", messages=MSG)   # a's circuit is open
    assert raw.headers["x-gateway-provider"] == "b" and raw.headers["x-gateway-attempts"] == "1"
    assert sorted(r.attempts for r in ledger(db)) == [1, 2, 2]
