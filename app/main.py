"""FastAPI application entrypoint.

Wires CORS, the startup lifespan (DB table creation), and feature routers.
The lifespan is resilient: if the database is unreachable at startup (e.g.
Postgres not running yet) the app still boots so the health check works.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")


async def _setup_checkpointer(app: FastAPI):
    """Create the LangGraph checkpointer (agent memory).

    Prefers Postgres so memory survives restarts; falls back to an in-memory
    saver if Postgres isn't reachable, so chat still works in dev.
    """
    from langgraph.checkpoint.memory import MemorySaver

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        pool = AsyncConnectionPool(
            conninfo=settings.checkpoint_db_url,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        app.state._checkpointer_pool = pool
        logger.info("Postgres checkpointer ready.")
        return saver
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Postgres checkpointer unavailable (%s); using in-memory saver.", exc
        )
        return MemorySaver()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup. Tolerate a missing DB so the app can still boot.
    try:
        from app.database import init_models

        await init_models()
        logger.info("Database tables ready.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database init skipped (%s). Is Postgres running?", exc)

    app.state.checkpointer = await _setup_checkpointer(app)
    yield

    pool = getattr(app.state, "_checkpointer_pool", None)
    if pool is not None:
        await pool.close()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


# Feature routers (each guarded so the app boots even mid-build)
for _module, _attr in [
    ("app.auth.router", "router"),
    ("app.documents.router", "router"),
    ("app.chat.router", "router"),
]:
    try:
        _mod = __import__(_module, fromlist=[_attr])
        app.include_router(getattr(_mod, _attr))
        logger.info("Mounted router: %s", _module)
    except ImportError:
        logger.info("Router not available yet: %s", _module)
