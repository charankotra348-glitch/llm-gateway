"""Tests run against SQLite by default and against Postgres when TEST_DATABASE_URL is set (CI does both).
Every test database is built by the real migrations, so they are tested too."""

import asyncio
import os

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from gateway.db import make_engine, make_sessions
from gateway.keys import create_key
from gateway.main import Gateway, create_app
from gateway.pricing import PriceTable
from gateway.providers.fake import FakeProvider
from gateway.resilience import Resilience, ResiliencePolicy
from gateway.routes import RouteTable


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _migrate(url: str) -> None:
    cfg = Config("alembic.ini")
    cfg.attributes["database_url"] = url
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")


async def _execute(url: str, *statements: str) -> None:
    engine = make_engine(url)
    async with engine.begin() as conn:
        for sql in statements:
            await conn.execute(text(sql))
    await engine.dispose()


@pytest.fixture(scope="session")
def postgres_url():
    """Postgres from TEST_DATABASE_URL: wiped and migrated from scratch once per run."""
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        asyncio.run(_execute(url, "DROP SCHEMA public CASCADE", "CREATE SCHEMA public"))
        _migrate(url)
    return url


@pytest.fixture
def database_url(tmp_path, postgres_url):
    if postgres_url:
        yield postgres_url
        asyncio.run(_execute(postgres_url, "TRUNCATE requests, api_keys"))
        return
    url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    _migrate(url)
    yield url


@pytest.fixture
def fake():
    return FakeProvider()


@pytest.fixture
def prices():
    return PriceTable.from_dict({"prices": {"fake": {"fake-echo": {"input_per_mtok": 1_000_000,
                                                                   "output_per_mtok": 2_000_000}}}})


@pytest.fixture
def gateway(database_url, fake, prices):
    routes = RouteTable.from_dict({"routes": {"demo": [{"provider": "fake", "model": "fake-echo"}]}})
    engine = make_engine(database_url)
    resilience = Resilience(ResiliencePolicy(attempt_timeout_s=1.0, backoff_s=0))
    return Gateway({"fake": fake}, routes, make_sessions(engine), prices=prices, resilience=resilience, engine=engine)


def run_db(database_url, fn):
    """Run `await fn(session)` on its own engine and event loop (the app's engine belongs to the app's loop)."""
    async def go():
        engine = make_engine(database_url)
        try:
            async with make_sessions(engine)() as session:
                return await fn(session)
        finally:
            await engine.dispose()
    return asyncio.run(go())


@pytest.fixture
def db(database_url):
    return lambda fn: run_db(database_url, fn)


@pytest.fixture
def make_key(db):
    """Create a key in the test database: make_key(budget_usd=None, rpm=None) -> (row, key)."""
    return lambda name="test", budget_usd=None, rpm=None: db(lambda s: create_key(s, name, budget_usd, rpm))


@pytest.fixture
def client(gateway):
    with TestClient(create_app(gateway)) as c:
        yield c


@pytest.fixture
def api_key(make_key):
    return make_key()[1]
