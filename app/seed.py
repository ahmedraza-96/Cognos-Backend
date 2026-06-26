"""Seed a public demo user for the demo login.

Idempotent: safe to run repeatedly. Creates the demo user if missing, or
resets its password to the known demo value if it already exists. Reuses the
app's own bcrypt hashing so the stored hash always matches what /auth/login
verifies against.

Credentials default to the values shown on the frontend login page and can be
overridden with env vars DEMO_EMAIL / DEMO_PASSWORD.

Run inside the running backend container:
    docker compose -f docker-compose.prod.yml exec backend python -m app.seed
"""
import asyncio
import os

from sqlalchemy import select

from app.auth.security import hash_password
from app.database import AsyncSessionLocal, init_models
from app.models import User

DEMO_EMAIL = os.getenv("DEMO_EMAIL", "demo@cognos.ai")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demo12345")


async def seed_demo_user() -> None:
    await init_models()  # ensure tables exist before we touch them
    async with AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == DEMO_EMAIL))
        if user is None:
            db.add(User(email=DEMO_EMAIL, hashed_password=hash_password(DEMO_PASSWORD)))
            action = "created"
        else:
            user.hashed_password = hash_password(DEMO_PASSWORD)
            action = "updated (password reset)"
        await db.commit()
    print(f"Demo user {action}: {DEMO_EMAIL} / {DEMO_PASSWORD}")


if __name__ == "__main__":
    asyncio.run(seed_demo_user())
