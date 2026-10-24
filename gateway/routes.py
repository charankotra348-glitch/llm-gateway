"""Routes: the model names clients ask for, and which provider models serve them.

    routes:
      fast:                       # what clients send as "model"
        - provider: gemini
          model: gemini-2.5-flash
        - provider: bedrock       # tried next if gemini fails
          model: us.amazon.nova-2-lite-v1:0
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from gateway.errors import ModelNotFound


class Target(BaseModel):
    provider: str
    model: str


class Route(BaseModel):
    name: str
    targets: list[Target] = Field(min_length=1)


class RouteTable:
    def __init__(self, routes: dict[str, Route]):
        self._routes = routes

    @classmethod
    def from_dict(cls, data: dict, *, known_providers: set[str] | None = None) -> "RouteTable":
        raw = (data or {}).get("routes")
        if not isinstance(raw, dict) or not raw:
            raise ValueError("routes file needs a non-empty 'routes:' mapping")
        routes = {}
        for name, targets in raw.items():
            try:
                route = Route(name=str(name), targets=targets)
            except ValidationError as exc:
                raise ValueError(f"route {name!r}: {exc.errors()[0]['msg']}") from exc
            if known_providers is not None:
                for t in route.targets:
                    if t.provider not in known_providers:
                        raise ValueError(f"route {name!r} uses provider {t.provider!r}, which is not "
                                         f"configured (configured: {sorted(known_providers)})")
            routes[route.name] = route
        return cls(routes)

    @classmethod
    def load(cls, path: str | Path, *, known_providers: set[str] | None = None) -> "RouteTable":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f), known_providers=known_providers)

    def get(self, name: str) -> Route:
        try:
            return self._routes[name]
        except KeyError:
            raise ModelNotFound(f"The model '{name}' does not exist. Available: "
                                f"{', '.join(sorted(self._routes))}") from None

    def names(self) -> list[str]:
        return sorted(self._routes)
