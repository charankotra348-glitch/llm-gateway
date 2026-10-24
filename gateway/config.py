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

    default_rpm: int = 60                    # requests per minute for keys without their own limit; 0 = none
    cache_ttl_s: float = 3600                # response cache (temperature-0 requests only); 0 = off
    cache_max_entries: int = 10_000

    redact_pii: bool = True                  # replace personal data and secrets before prompts reach a provider

    key_cache_ttl_s: float = 30              # how long a valid key is trusted without a lookup; 0 = always look up
    ledger_batched: bool = True              # write ledger rows in the background instead of before responding

    gemini_api_key: str = ""
    bedrock_region: str = ""          # empty = Bedrock not configured
    ollama_url: str = ""              # e.g. http://localhost:11434; empty = not configured
    enable_fake: bool = True          # in-process fakes for demos and tests: "fake" and "flaky"
    flaky_failure_rate: float = 0.5   # share of calls the "flaky" fake fails, to demo fallback
    fake_chunk_delay_s: float = 0.05  # pause between streamed words from the fakes, so streaming is visible


@lru_cache
def get_settings() -> Settings:
    return Settings()
