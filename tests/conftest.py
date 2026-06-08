"""Shared test fixtures.

Tests run against an in-memory SQLite database (our models are DB-agnostic),
so the suite needs neither Postgres nor Docker. The `get_db` dependency is
overridden to hand out sessions bound to that test database.
"""
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_engine():
    engine = create_async_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    import app.models  # noqa: F401  (register tables on Base.metadata)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def client(test_engine):
    import shutil
    import tempfile

    from fastapi.testclient import TestClient
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from app.rag import deps as rag_deps

    TestSession = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def override_get_db():
        async with TestSession() as session:
            yield session

    tmpdir = tempfile.mkdtemp()
    fake = DeterministicFakeEmbedding(size=64)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[rag_deps.embeddings_provider] = lambda: fake
    app.dependency_overrides[rag_deps.chroma_dir_provider] = lambda: f"{tmpdir}/chroma"
    app.dependency_overrides[rag_deps.upload_dir_provider] = lambda: f"{tmpdir}/uploads"

    # No `with` block: skip the lifespan so tests don't touch the real Postgres engine.
    yield TestClient(app)

    app.dependency_overrides.clear()
    shutil.rmtree(tmpdir, ignore_errors=True)
