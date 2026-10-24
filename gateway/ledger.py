"""The request ledger: what every request cost, and how much of its budget a key has used."""

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models import RequestLog


def month_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


@dataclass
class Usage:
    spend_micros: int
    requests: int
    unpriced_requests: int        # succeeded but could not be priced, so not counted in spend


async def month_usage(session: AsyncSession, key_id: str, now: datetime | None = None) -> Usage:
    since = month_start(now)
    spend, total, unpriced = (await session.execute(
        select(
            func.coalesce(func.sum(RequestLog.cost_micros), 0),
            func.count(),
            func.count().filter(RequestLog.cost_micros.is_(None), RequestLog.outcome == "ok"),
        ).where(RequestLog.key_id == key_id, RequestLog.created_at >= since)
    )).one()
    return Usage(int(spend), int(total), int(unpriced))
