"""API keys: `gw_` + 43 random characters. Only a SHA-256 of the key is stored, so a database
leak does not leak usable keys. The key is shown once, at creation.

    python -m gateway.keys create --name my-app --budget 5 --rpm 30   # USD a month (UTC), requests a minute
    python -m gateway.keys list
    python -m gateway.keys revoke <prefix>
"""

import argparse
import asyncio
import hashlib
import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gateway.models import ApiKey

PREFIX = "gw_"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> str:
    return PREFIX + secrets.token_urlsafe(32)


async def create_key(session: AsyncSession, name: str, budget_usd: float | None = None,
                     rpm: int | None = None) -> tuple[ApiKey, str]:
    key = new_key()
    row = ApiKey(name=name, prefix=key[: len(PREFIX) + 6], key_hash=hash_key(key),
                 monthly_budget_micros=None if budget_usd is None else round(budget_usd * 1_000_000),
                 rate_limit_rpm=rpm)
    session.add(row)
    await session.commit()
    return row, key


async def find_key(session: AsyncSession, key: str) -> ApiKey | None:
    if not key.startswith(PREFIX):
        return None
    row = await session.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(key)))
    if row is None or row.revoked_at is not None:
        return None
    return row


class KeyCache:
    """Remembers valid keys for `ttl_s` seconds, so most requests skip the database lookup.

    The cost: a key revoked (or given a new budget or rate limit) keeps its old state in a running
    gateway for up to `ttl_s`. Unknown keys are never cached, so a new key works at once.
    `ttl_s=0` turns caching off."""

    def __init__(self, ttl_s: float = 30, *, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._entries: dict[str, tuple[float, ApiKey]] = {}      # key hash -> (expires at, row)

    async def find(self, sessions: async_sessionmaker, key: str) -> ApiKey | None:
        digest = hash_key(key)
        if self.ttl_s > 0 and (entry := self._entries.get(digest)) and self._clock() < entry[0]:
            return entry[1]
        async with sessions() as session:
            row = await find_key(session, key)
        if row is None:
            self._entries.pop(digest, None)
        elif self.ttl_s > 0:
            if len(self._entries) >= self.max_entries:
                self._entries.pop(next(iter(self._entries)))       # oldest inserted
            self._entries[digest] = (self._clock() + self.ttl_s, row)
        return row


async def _main(argv=None) -> None:
    from gateway.config import get_settings
    from gateway.db import make_engine, make_sessions

    parser = argparse.ArgumentParser(prog="python -m gateway.keys")
    sub = parser.add_subparsers(dest="cmd", required=True)
    create = sub.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--budget", type=float, help="USD per calendar month (UTC); omit for no limit")
    create.add_argument("--rpm", type=int, help="requests per minute; omit for the gateway default, 0 for none")
    sub.add_parser("list")
    revoke = sub.add_parser("revoke")
    revoke.add_argument("prefix")
    args = parser.parse_args(argv)

    engine = make_engine(get_settings().database_url)
    try:
        async with make_sessions(engine)() as session:
            if args.cmd == "create":
                row, key = await create_key(session, args.name, args.budget, args.rpm)
                print(f"Created {row.name} ({row.prefix}...). This is the only time the key is shown:\n\n  {key}\n")
            elif args.cmd == "list":
                for row in await session.scalars(select(ApiKey).order_by(ApiKey.created_at)):
                    budget = "no limit" if row.monthly_budget_micros is None else \
                        f"${row.monthly_budget_micros / 1e6:.2f}/month"
                    rpm = "default rpm" if row.rate_limit_rpm is None else \
                        ("no rpm limit" if row.rate_limit_rpm == 0 else f"{row.rate_limit_rpm} rpm")
                    state = "revoked" if row.revoked_at else "active"
                    print(f"{row.prefix}...  {row.name:<24} {budget:<16} {rpm:<14} {state}")
            else:
                rows = list(await session.scalars(select(ApiKey).where(ApiKey.prefix == args.prefix)))
                if len(rows) != 1:
                    raise SystemExit(f"{len(rows)} keys have prefix {args.prefix!r}; expected exactly one")
                rows[0].revoked_at = datetime.now(UTC)
                await session.commit()
                print(f"Revoked {rows[0].name} ({rows[0].prefix}...)")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
