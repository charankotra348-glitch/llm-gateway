from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///gateway.db"   # production: postgresql+asyncpg://...
    routes_file: str = "routes.yaml"
    prices_file: str = "prices.yaml"
    request_timeout_s: float = 30.0

    gemini_api_key: str = ""
    bedrock_region: str = ""          # empty = Bedrock not configured
    ollama_url: str = ""              # e.g. http://localhost:11434; empty = not configured
    enable_fake: bool = True          # the in-process fake provider, for demos and tests


@lru_cache
def get_settings() -> Settings:
    return Settings()
