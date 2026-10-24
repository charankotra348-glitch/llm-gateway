import pytest

from gateway.db import normalize_database_url


@pytest.mark.parametrize("given, expected", [
    ("postgresql://u:p@ep-x.neon.tech/db?sslmode=require&channel_binding=require",
     "postgresql+asyncpg://u:p@ep-x.neon.tech/db?ssl=require"),
    ("postgres://u:p@host:5432/db", "postgresql+asyncpg://u:p@host:5432/db"),
    ("postgresql+asyncpg://u:p@host/db", "postgresql+asyncpg://u:p@host/db"),
    ("postgresql://u:p%40ss@host/db?sslmode=verify-full", "postgresql+asyncpg://u:p%40ss@host/db?ssl=verify-full"),
    ("sqlite+aiosqlite:///gateway.db", "sqlite+aiosqlite:///gateway.db"),
])
def test_normalize(given, expected):
    assert normalize_database_url(given) == expected
