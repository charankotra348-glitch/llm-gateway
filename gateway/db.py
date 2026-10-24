from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def normalize_database_url(url: str) -> str:
    """Accept the Postgres URL a host hands out and turn it into one the async driver understands.

    Hosts like Neon, Render and Heroku give `postgres://` or `postgresql://...?sslmode=require&channel_binding=require`.
    The gateway needs the asyncpg driver, which takes `ssl=` instead of `sslmode=` and has no
    `channel_binding` option, so both are translated or dropped. Other URLs pass through untouched.
    """
    parsed = make_url(url)
    if parsed.drivername not in ("postgres", "postgresql", "postgresql+asyncpg"):
        return url
    query = dict(parsed.query)
    if "sslmode" in query:
        query.setdefault("ssl", query.pop("sslmode"))
    query.pop("channel_binding", None)
    return parsed.set(drivername="postgresql+asyncpg", query=query).render_as_string(hide_password=False)


def make_engine(url: str) -> AsyncEngine:
    # pool_pre_ping: a serverless database (Neon) closes idle connections; check before reuse
    return create_async_engine(normalize_database_url(url), pool_pre_ping=True)


def make_sessions(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)
