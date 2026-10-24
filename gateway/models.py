"""Tables: api_keys and requests (the ledger).

Money is stored as integer micro-dollars (1 USD = 1_000_000) so sums are exact on every database.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from gateway.db import Base


def _now() -> datetime:
    return datetime.now(UTC)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    name: Mapped[str] = mapped_column(String(100))
    prefix: Mapped[str] = mapped_column(String(16))                      # shown in lists and logs
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)       # sha256 of the key; the key is never stored
    monthly_budget_micros: Mapped[int | None] = mapped_column(BigInteger, nullable=True)   # None = no limit
    rate_limit_rpm: Mapped[int | None] = mapped_column(Integer, nullable=True)   # None = gateway default; 0 = none
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RequestLog(Base):
    """One row per chat completion the gateway accepted, successful or not."""

    __tablename__ = "requests"
    __table_args__ = (Index("ix_requests_key_created", "key_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(48), primary_key=True)        # the chatcmpl-... id returned
    key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id"))
    route: Mapped[str] = mapped_column(String(100))
    provider: Mapped[str] = mapped_column(String(20))       # the one that answered, or the last one tried
    model: Mapped[str] = mapped_column(String(100))
    outcome: Mapped[str] = mapped_column(String(40))                     # ok | an error code
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_micros: Mapped[int | None] = mapped_column(BigInteger, nullable=True)   # None = not priced
    latency_ms: Mapped[float] = mapped_column(Float)
    attempts: Mapped[int] = mapped_column(Integer, default=1, server_default="1")   # provider calls made
    redactions: Mapped[int] = mapped_column(Integer, default=0, server_default="0")  # values hidden from provider
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
