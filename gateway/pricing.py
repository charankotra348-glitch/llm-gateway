"""Estimated cost from the provider's own token counts and a price table (prices.yaml).

An estimate, not an invoice: list prices only (no free tiers, discounts or cached-input rates),
and no cost at all when either the price or the token counts are unknown.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Price(BaseModel):
    input_per_mtok: float = Field(ge=0)     # USD per 1M input tokens
    output_per_mtok: float = Field(ge=0)


class PriceTable:
    def __init__(self, prices: dict[tuple[str, str], Price], free_providers: set[str] = frozenset()):
        self._prices = prices
        self._free = set(free_providers)

    @classmethod
    def from_dict(cls, data: dict) -> "PriceTable":
        prices = {}
        for provider, models in (data.get("prices") or {}).items():
            for model, price in (models or {}).items():
                prices[(provider, model)] = Price(**price)
        return cls(prices, set(data.get("free_providers") or []))

    @classmethod
    def load(cls, path: str | Path) -> "PriceTable":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def knows(self, provider: str, model: str) -> bool:
        return provider in self._free or (provider, model) in self._prices

    def cost_micros(self, provider: str, model: str, input_tokens: int | None,
                    output_tokens: int | None) -> int | None:
        if input_tokens is None or output_tokens is None:
            return None
        if provider in self._free:
            return 0
        price = self._prices.get((provider, model))
        if price is None:
            return None
        # USD per 1M tokens == micro-dollars per token
        return round(input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok)
