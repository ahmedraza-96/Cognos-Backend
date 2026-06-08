"""Async SQLAlchemy engine, session factory, and declarative Base.

Consumers get a session via the `get_db` FastAPI dependency. Keeping the engine
here means swapping the database is a single config change (DATABASE_URL).
"""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    pass


engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a database session."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_models() -> None:
    """Create tables for all imported models. Called on startup.

    Imports models for side effects so their tables register on Base.metadata.
    """
    import app.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
