"""Per-key rate limits and the response cache."""

import openai
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from gateway.cache import ResponseCache, plan
from gateway.db import make_engine, make_sessions
from gateway.errors import ProviderUnavailable
from gateway.limits import RateLimiter, TokenBucket
from gateway.main import Gateway, create_app
from gateway.models import RequestLog
from gateway.pricing import PriceTable
from gateway.providers.fake import FakeProvider
from gateway.resilience import Resilience, ResiliencePolicy
from gateway.routes import RouteTable, Target
from gateway.schemas import ChatRequest, Completion, Message

MSG = [{"role": "user", "content": "hello there"}]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# --------------------------------------------------------------------------- token bucket
def test_bucket_allows_a_burst_then_refills_at_the_rate():
    clock = Clock()
    b = TokenBucket(rpm=3, clock=clock)
    assert [b.take().remaining for _ in range(3)] == [2, 1, 0]
    denied = b.take()
    assert not denied.allowed and denied.retry_after_s == pytest.approx(20)    # 60 s / 3
    clock.now += 20
    assert b.take().allowed
    assert not b.take().allowed


def test_bucket_never_holds_more_than_its_size():
    clock = Clock()
    b = TokenBucket(rpm=2, clock=clock)
    clock.now += 3600
    assert [b.take().allowed for _ in range(3)] == [True, True, False]


def test_limiter_defaults_unlimited_and_limit_changes():
    clock = Clock()
    limiter = RateLimiter(default_rpm=1, clock=clock)
    assert limiter.check("k", None).allowed and not limiter.check("k", None).allowed
    assert limiter.check("k", 0) is None                       # 0 = this key is unlimited
    assert limiter.check("k", 5).allowed                       # a new limit starts a fresh bucket
    assert RateLimiter(default_rpm=0).check("k", None) is None
    assert limiter.check("other", None).allowed                # buckets are per key


# --------------------------------------------------------------------------- cache unit
CHAT = ChatRequest(messages=[Message(role="user", text="hi")], temperature=0)
TARGET = Target(provider="fake", model="fake-echo")


def test_only_temperature_zero_is_cacheable():
    assert plan("k", "demo", CHAT, None).read
    for t in (None, 0.7):
        p = plan("k", "demo", CHAT.model_copy(update={"temperature": t}), None)
        assert (p.key, p.reason) == (None, "bypass")


def test_cache_control_directives():
    assert plan("k", "demo", CHAT, "no-store").key is None
    p = plan("k", "demo", CHAT, "max-age=0, No-Cache")
    assert p.key and not p.read and p.write


def test_cache_key_covers_key_route_and_every_option():
    base = plan("k", "demo", CHAT, None).key
    assert plan("k2", "demo", CHAT, None).key != base
    assert plan("k", "other", CHAT, None).key != base
    for change in ({"max_tokens": 5}, {"stop": ["x"]}, {"json_mode": True}, {"top_p": 0.5},
                   {"messages": [Message(role="user", text="hi!")]}):
        assert plan("k", "demo", CHAT.model_copy(update=change), None).key != base


def test_cache_ttl_and_lru():
    clock = Clock()
    cache = ResponseCache(ttl_s=60, max_entries=2, clock=clock)
    answer = (Completion(text="a", model="m"), TARGET)
    cache.put("1", answer)
    cache.put("2", answer)
    cache.get("1")                       # 1 is now the most recently used
    cache.put("3", answer)               # evicts 2
    assert cache.get("2") is None and cache.get("1") and cache.get("3")
    clock.now += 60
    assert cache.get("1") is None and len(cache) == 1


def test_disabled_cache_stores_nothing():
    cache = ResponseCache(ttl_s=0)
    cache.put("1", (Completion(text="a", model="m"), TARGET))
    assert not cache.enabled and len(cache) == 0


# --------------------------------------------------------------------------- through the API
@pytest.fixture
def app_with(database_url, make_key):
    """make(rpm=None default, cache=True) -> (sdk_for(key), client, fake, key)."""
    def make(default_rpm=0, cache=True):
        fake = FakeProvider()
        routes = RouteTable.from_dict({"routes": {"demo": [{"provider": "fake", "model": "fake-echo"}]}})
        engine = make_engine(database_url)
        gw = Gateway({"fake": fake}, routes, make_sessions(engine),
                     prices=PriceTable.from_dict({"prices": {"fake": {"fake-echo": {"input_per_mtok": 1_000_000,
                                                                                    "output_per_mtok": 0}}}}),
                     resilience=Resilience(ResiliencePolicy(backoff_s=0)), limiter=RateLimiter(default_rpm),
                     cache=ResponseCache(ttl_s=3600 if cache else 0), engine=engine)
        client = TestClient(create_app(gw)).__enter__()
        made.append(client)

        def sdk(key, retries=0):
            return openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=client,
                                 max_retries=retries)
        return sdk, client, fake
    made = []
    yield make
    for c in made:
        c.__exit__(None, None, None)


def rows(db):
    async def q(s):
        return list(await s.scalars(select(RequestLog).order_by(RequestLog.created_at)))
    return db(q)


def test_rate_limit_returns_429_with_retry_after(app_with, make_key):
    sdk, _, _ = app_with()
    _, key = make_key(rpm=2)
    raw = sdk(key).chat.completions.with_raw_response.create(model="demo", messages=MSG)
    assert raw.headers["x-ratelimit-limit-requests"] == "2" and raw.headers["x-ratelimit-remaining-requests"] == "1"
    sdk(key).chat.completions.create(model="demo", messages=MSG)
    with pytest.raises(openai.RateLimitError) as exc:
        sdk(key).chat.completions.create(model="demo", messages=MSG)
    assert exc.value.code == "rate_limit_exceeded"
    h = exc.value.response.headers
    assert h["x-should-retry"] == "true" and 25 <= int(h["retry-after"]) <= 31 and int(h["retry-after-ms"]) > 0


def test_rate_limit_is_per_key_and_default_applies(app_with, make_key):
    sdk, _, _ = app_with(default_rpm=1)
    _, a = make_key()
    _, b = make_key()
    _, free = make_key(rpm=0)
    sdk(a).chat.completions.create(model="demo", messages=MSG)
    with pytest.raises(openai.RateLimitError):
        sdk(a).chat.completions.create(model="demo", messages=MSG)
    sdk(b).chat.completions.create(model="demo", messages=MSG)
    for _ in range(5):
        sdk(free).chat.completions.create(model="demo", messages=MSG)


def test_rate_limited_requests_are_not_recorded(app_with, make_key, db):
    sdk, _, fake = app_with()
    _, key = make_key(rpm=1)
    sdk(key).chat.completions.create(model="demo", messages=MSG)
    with pytest.raises(openai.RateLimitError):
        sdk(key).chat.completions.create(model="demo", messages=MSG)
    assert len(fake.calls) == 1 and len(rows(db)) == 1


def test_identical_temperature_zero_request_is_served_from_cache(app_with, api_key, db):
    sdk, client, fake = app_with()
    first = sdk(api_key).chat.completions.with_raw_response.create(model="demo", messages=MSG, temperature=0)
    second = sdk(api_key).chat.completions.with_raw_response.create(model="demo", messages=MSG, temperature=0)
    assert first.headers["x-gateway-cache"] == "miss" and second.headers["x-gateway-cache"] == "hit"
    assert second.headers["x-gateway-provider"] == "cache" and second.headers["x-gateway-attempts"] == "0"
    a, b = first.parse(), second.parse()
    assert a.choices[0].message.content == b.choices[0].message.content and a.id != b.id
    assert len(fake.calls) == 1
    hit = rows(db)[1]
    assert (hit.provider, hit.model, hit.cost_micros, hit.attempts) == ("cache", "fake-echo", 0, 0)
    assert rows(db)[0].cost_micros > 0
    usage = client.get("/v1/usage", headers={"authorization": f"Bearer {api_key}"}).json()
    assert usage["cache_hits"] == 1 and usage["requests"] == 2


def test_sampling_requests_are_never_cached(app_with, api_key):
    sdk, _, fake = app_with()
    for t in (None, 0.7):
        for _ in range(2):
            kw = {} if t is None else {"temperature": t}
            raw = sdk(api_key).chat.completions.with_raw_response.create(model="demo", messages=MSG, **kw)
            assert raw.headers["x-gateway-cache"] == "bypass"
    assert len(fake.calls) == 4


def test_cache_is_scoped_to_the_key(app_with, make_key):
    sdk, _, fake = app_with()
    _, a = make_key()
    _, b = make_key()
    sdk(a).chat.completions.create(model="demo", messages=MSG, temperature=0)
    raw = sdk(b).chat.completions.with_raw_response.create(model="demo", messages=MSG, temperature=0)
    assert raw.headers["x-gateway-cache"] == "miss" and len(fake.calls) == 2


def test_cache_control_headers(app_with, api_key):
    sdk, _, fake = app_with()
    c = sdk(api_key).chat.completions
    c.create(model="demo", messages=MSG, temperature=0, extra_headers={"cache-control": "no-store"})
    assert c.with_raw_response.create(model="demo", messages=MSG, temperature=0).headers["x-gateway-cache"] == "miss"
    raw = c.with_raw_response.create(model="demo", messages=MSG, temperature=0,
                                     extra_headers={"cache-control": "no-cache"})
    assert raw.headers["x-gateway-cache"] == "no-cache" and len(fake.calls) == 3


def test_failures_are_not_cached(app_with, api_key):
    sdk, _, fake = app_with()
    fake.script = [ProviderUnavailable("down")] * 2
    with pytest.raises(openai.APIStatusError):
        sdk(api_key).chat.completions.create(model="demo", messages=MSG, temperature=0)
    raw = sdk(api_key).chat.completions.with_raw_response.create(model="demo", messages=MSG, temperature=0)
    assert raw.headers["x-gateway-cache"] == "miss"


def test_cache_hits_are_served_even_when_the_budget_is_spent(app_with, make_key):
    sdk, _, _ = app_with()
    _, key = make_key(budget_usd=1)          # test price: $1 per input token, so one request spends it
    sdk(key).chat.completions.create(model="demo", messages=MSG, temperature=0)
    sdk(key).chat.completions.create(model="demo", messages=MSG, temperature=0)      # hit: free
    with pytest.raises(openai.RateLimitError) as exc:
        sdk(key).chat.completions.create(model="demo", messages=[{"role": "user", "content": "new"}], temperature=0)
    assert exc.value.code == "budget_exceeded"


def test_cache_off_means_no_header(app_with, api_key):
    sdk, _, _ = app_with(cache=False)
    raw = sdk(api_key).chat.completions.with_raw_response.create(model="demo", messages=MSG, temperature=0)
    assert "x-gateway-cache" not in raw.headers


def test_sdk_waits_out_a_rate_limit_by_itself(app_with, make_key, monkeypatch):
    """The OpenAI SDK reads retry-after and retries: a client with retries on never sees the 429."""
    sdk, _, fake = app_with()
    _, key = make_key(rpm=60)                # one token a second
    client = sdk(key, retries=2)
    for _ in range(60):
        client.chat.completions.create(model="demo", messages=MSG)
    client.chat.completions.create(model="demo", messages=MSG)      # 61st: SDK waits ~1 s, then succeeds
    assert len(fake.calls) == 61
