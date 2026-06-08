"""Auth dependencies: resolve the current user from a Bearer JWT."""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import TokenError, decode_access_token
from app.database import get_db
from app.models import User

_bearer = HTTPBearer(auto_error=True)

_credentials_exc = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    try:
        payload = decode_access_token(credentials.credentials)
    except TokenError:
        raise _credentials_exc

    user_id = payload.get("sub")
    if not user_id:
        raise _credentials_exc

    user = await db.get(User, user_id)
    if user is None:
        raise _credentials_exc
    return user
