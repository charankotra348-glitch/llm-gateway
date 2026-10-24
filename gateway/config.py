from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///gateway.db"   # production: postgresql+asyncpg://...
    routes_file: str = "routes.yaml"
    prices_file: str = "prices.yaml"
    request_timeout_s: float = 30.0          # one call to one provider
    total_timeout_s: float = 60.0            # a whole request, across retries and fallbacks
    attempts_per_target: int = 2             # tries on a target that is down (5xx) before moving on
    retry_backoff_s: float = 0.2
    breaker_threshold: int = 5               # consecutive failures that open a target's circuit
    breaker_cooldown_s: float = 30.0

    gemini_api_key: str = ""
    bedrock_region: str = ""          # empty = Bedrock not configured
    ollama_url: str = ""              # e.g. http://localhost:11434; empty = not configured
    enable_fake: bool = True          # in-process fakes for demos and tests: "fake" and "flaky"
    flaky_failure_rate: float = 0.5   # share of calls the "flaky" fake fails, to demo fallback


@lru_cache
def get_settings() -> Settings:
    return Settings()
