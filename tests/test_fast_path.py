"""The two things that keep the database off the request path: the key cache and the batched ledger."""

import asyncio
from datetime import UTC, datetime

import openai
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from gateway.db import make_engine, make_sessions
from gateway.keys import KeyCache, create_key
from gateway.ledger import LedgerWriter
from gateway.main import Gateway, create_app
from gateway.models import ApiKey, RequestLog
from gateway.pricing import PriceTable
from gateway.providers.fake import FakeProvider
from gateway.routes import RouteTable

MSG = [{"role": "user", "content": "hi"}]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class CountingSessions:
    """Wraps a sessionmaker and counts how many sessions were opened."""

    def __init__(self, sessions):
        self.sessions, self.opened = sessions, 0

    def __call__(self):
        self.opened += 1
        return self.sessions()


def row(n: int, key_id: str) -> RequestLog:
    return RequestLog(id=f"chatcmpl-{n}", key_id=key_id, route="demo", provider="fake", model="m", outcome="ok",
                      latency_ms=1.0, created_at=datetime.now(UTC))


async def new_key(sessions) -> tuple[ApiKey, str]:
    async with sessions() as session:
        return await create_key(session, "test")


async def count_in(engine) -> int:
    async with make_sessions(engine)() as s:
        return await s.scalar(select(func.count()).select_from(RequestLog))


def count_rows(db) -> int:
    async def q(s):
        return await s.scalar(select(func.count()).select_from(RequestLog))
    return db(q)


# --------------------------------------------------------------------------- key cache
@pytest.mark.anyio
async def test_key_cache_skips_the_database_until_the_ttl(database_url):
    engine = make_engine(database_url)
    _, key = await new_key(make_sessions(engine))
    sessions = CountingSessions(make_sessions(engine))
    clock = Clock()
    cache = KeyCache(ttl_s=30, clock=clock)
    try:
        first = await cache.find(sessions, key)
        assert first is not None and sessions.opened == 1
        for _ in range(5):
            assert (await cache.find(sessions, key)).id == first.id
        assert sessions.opened == 1
        clock.now += 30
        await cache.find(sessions, key)
        assert sessions.opened == 2
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_unknown_keys_are_never_cached(database_url):
    engine = make_engine(database_url)
    sessions = CountingSessions(make_sessions(engine))
    cache = KeyCache(ttl_s=30)
    try:
        for _ in range(3):
            assert await cache.find(sessions, "gw_nope") is None
        assert sessions.opened == 3
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_revocation_takes_effect_after_the_ttl(database_url):
    engine = make_engine(database_url)
    row_, key = await new_key(make_sessions(engine))
    clock = Clock()
    cache = KeyCache(ttl_s=30, clock=clock)
    try:
        assert await cache.find(make_sessions(engine), key)

        async with make_sessions(engine)() as s:
            (await s.get(ApiKey, row_.id)).revoked_at = datetime.now(UTC)
            await s.commit()
        assert await cache.find(make_sessions(engine), key)          # the documented window
        clock.now += 30
        assert await cache.find(make_sessions(engine), key) is None
    finally:
        await engine.dispose()


def test_key_cache_is_bounded():
    cache = KeyCache(ttl_s=30, max_entries=2)
    for i in range(3):
        if len(cache._entries) >= cache.max_entries:
            cache._entries.pop(next(iter(cache._entries)))
        cache._entries[str(i)] = (1e9, None)
    assert list(cache._entries) == ["1", "2"]


# --------------------------------------------------------------------------- batched ledger
@pytest.mark.anyio
async def test_batched_writer_writes_everything_and_batches_under_load(database_url):
    engine = make_engine(database_url)
    key_row, _ = await new_key(make_sessions(engine))
    sessions = CountingSessions(make_sessions(engine))
    writer = LedgerWriter(sessions, batched=True, max_batch=100)
    await writer.start()
    try:
        await asyncio.gather(*(writer.write(row(i, key_row.id)) for i in range(300)))
        await writer.flush()
        assert writer.written == 300 and writer.dropped == 0
        assert sessions.opened < 300 / 2           # many rows per transaction
        assert await count_in(engine) == 300
    finally:
        await writer.stop()
        await engine.dispose()


@pytest.mark.anyio
async def test_stop_drains_the_queue(database_url):
    engine = make_engine(database_url)
    key_row, _ = await new_key(make_sessions(engine))
    writer = LedgerWriter(make_sessions(engine), batched=True)
    await writer.start()
    for i in range(50):
        await writer.write(row(i, key_row.id))
    await writer.stop()
    assert await count_in(engine) == 50
    await engine.dispose()


@pytest.mark.anyio
async def test_a_failing_batch_is_retried_then_counted_as_dropped(caplog):
    class Broken:
        calls = 0

        def __call__(self):
            Broken.calls += 1
            raise ConnectionError("database is down")

    writer = LedgerWriter(Broken(), batched=True, retries=3, retry_delay_s=0)
    await writer.start()
    await writer.write(RequestLog(id="chatcmpl-x"))
    await writer.flush()
    await writer.stop()
    assert Broken.calls == 3 and writer.dropped == 1 and "ledger_write_failed" in caplog.text


@pytest.mark.anyio
async def test_full_queue_makes_writers_wait_instead_of_dropping():
    release = asyncio.Event()
    written = []

    class Slow:
        def __call__(self):
            return self

        async def __aenter__(self):
            await release.wait()
            return self

        async def __aexit__(self, *exc):
            return False

        def add_all(self, rows):
            written.extend(rows)

        async def commit(self):
            pass

    writer = LedgerWriter(Slow(), batched=True, max_queue=2, max_batch=1)
    await writer.start()
    for i in range(3):                       # 1 being written + 2 queued
        await writer.write(RequestLog(id=str(i)))
    blocked = asyncio.create_task(writer.write(RequestLog(id="3")))
    await asyncio.sleep(0.05)
    assert not blocked.done()                # backpressure, not a dropped row
    release.set()
    await blocked
    await writer.stop()
    assert [r.id for r in written] == ["0", "1", "2", "3"]


def test_direct_writer_writes_before_returning(database_url, make_key, db):
    key_row, _ = make_key()

    async def go():
        engine = make_engine(database_url)
        writer = LedgerWriter(make_sessions(engine))          # not started: direct
        await writer.write(row(1, key_row.id))
        await engine.dispose()
    asyncio.run(go())
    assert count_rows(db) == 1


# --------------------------------------------------------------------------- the app, production settings
def test_app_with_cached_keys_and_batched_ledger(database_url, make_key, db):
    _, key = make_key()
    engine = make_engine(database_url)
    sessions = CountingSessions(make_sessions(engine))
    gw = Gateway({"fake": FakeProvider()},
                 RouteTable.from_dict({"routes": {"demo": [{"provider": "fake", "model": "fake-echo"}]}}),
                 sessions, prices=PriceTable.from_dict({"free_providers": ["fake"]}),
                 keys=KeyCache(ttl_s=30), ledger=LedgerWriter(sessions, batched=True), engine=engine)
    with TestClient(create_app(gw)) as client:
        sdk = openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=client, max_retries=0)
        for _ in range(20):
            sdk.chat.completions.create(model="demo", messages=MSG)
        client.portal.call(gw.ledger.flush)
        assert count_rows(db) == 20
        assert sessions.opened == 1 + gw.ledger.batches      # one key lookup; every other session wrote rows
    assert gw.ledger.dropped == 0
