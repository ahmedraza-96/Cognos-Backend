"""Application configuration loaded from environment / .env.

Everything the app needs to run is centralized here so other modules depend on
a single, typed `settings` object rather than reading os.environ directly.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_name: str = "AI Agent Platform"
    environment: str = "development"
    cors_origins: str = "http://localhost:3000"

    # Auth
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # Database
    database_url: str = "postgresql+asyncpg://agent:agent@localhost:5432/agentdb"
    checkpoint_db_url: str = "postgresql://agent:agent@localhost:5432/agentdb"

    # OpenAI
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # Tavily (optional)
    tavily_api_key: str = ""

    # RAG / Chroma
    chroma_dir: str = "./data/chroma"

    # Uploads
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 10

    # HTTP tool SSRF allowlist
    http_tool_allowed_hosts: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_http_hosts(self) -> list[str]:
        return [h.strip().lower() for h in self.http_tool_allowed_hosts.split(",") if h.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
