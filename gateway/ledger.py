"""The request ledger: what every request cost, and how much of its budget a key has used."""

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gateway.models import RequestLog

CACHE_PROVIDER = "cache"        # the ledger's provider for requests answered from the response cache


def month_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


@dataclass
class Usage:
    spend_micros: int
    requests: int
    unpriced_requests: int        # succeeded but could not be priced, so not counted in spend
    cache_hits: int               # answered from the response cache, at no cost


async def month_usage(session: AsyncSession, key_id: str, now: datetime | None = None) -> Usage:
    since = month_start(now)
    spend, total, unpriced, hits = (await session.execute(
        select(
            func.coalesce(func.sum(RequestLog.cost_micros), 0),
            func.count(),
            func.count().filter(RequestLog.cost_micros.is_(None), RequestLog.outcome == "ok"),
            func.count().filter(RequestLog.provider == CACHE_PROVIDER),
        ).where(RequestLog.key_id == key_id, RequestLog.created_at >= since)
    )).one()
    return Usage(int(spend), int(total), int(unpriced), int(hits))


log = logging.getLogger("gateway")


class LedgerWriter:
    """Writes ledger rows, either before the request returns (direct) or in the background (batched).

    Batched is what production uses: a row goes on a queue and the response goes out without
    waiting for the database. A background task inserts whatever has queued up, in one transaction.
    Under load the queue fills while an insert runs, so batches grow by themselves, with no timer.

    Trade-offs, chosen on purpose:
    - Rows still queued when the process is killed outright are lost. A normal shutdown drains the queue.
    - If the database falls behind, the queue fills and requests wait to enqueue (backpressure)
      rather than rows being dropped silently.
    - A batch that keeps failing is retried, then logged and counted in `dropped`.
    - Budgets read the database, so rows still in the queue (milliseconds' worth) aren't counted yet.
      Budgets are already a soft limit.
    """

    def __init__(self, sessions: async_sessionmaker, *, batched: bool = False, max_batch: int = 500,
                 max_queue: int = 10_000, retries: int = 3, retry_delay_s: float = 0.5):
        self._sessions = sessions
        self.batched = batched
        self.max_batch = max_batch
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self._queue: asyncio.Queue[RequestLog] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self.written = 0
        self.dropped = 0
        self.batches = 0

    async def start(self) -> None:
        if self.batched and self._task is None:
            self._task = asyncio.create_task(self._run(), name="ledger-writer")

    async def stop(self) -> None:
        """Drain the queue, then stop the background task."""
        if self._task is None:
            return
        await self.flush()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def write(self, row: RequestLog) -> None:
        if self._task is None:
            await self._insert([row])          # direct: errors reach the caller
            return
        await self._queue.put(row)             # waits only when the queue is full

    async def flush(self) -> None:
        """Wait until every queued row has been written (or given up on)."""
        await self._queue.join()

    async def _run(self) -> None:
        while True:
            batch = [await self._queue.get()]
            while len(batch) < self.max_batch and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            try:
                await self._insert_with_retries(batch)
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def _insert_with_retries(self, batch: list[RequestLog]) -> None:
        for attempt in range(self.retries):
            try:
                await self._insert(batch)
                return
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 == self.retries:
                    self.dropped += len(batch)
                    log.error(json.dumps({"event": "ledger_write_failed", "rows": len(batch),
                                          "error": type(exc).__name__, "dropped_total": self.dropped}))
                    return
                await asyncio.sleep(self.retry_delay_s * 2 ** attempt)

    async def _insert(self, rows: list[RequestLog]) -> None:
        async with self._sessions() as session:
            session.add_all(rows)
            await session.commit()
        self.written += len(rows)
        self.batches += 1
