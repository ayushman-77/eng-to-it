"""
Security utilities — password hashing, JWT creation / verification,
and a FastAPI dependency that extracts the authenticated user from the
Authorization header.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings
from app.db.repositories import UserRepository

# ── Password Hashing ────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True when *plain* matches *hashed*."""
    return _pwd_context.verify(plain, hashed)


# ── JWT Tokens ───────────────────────────────────────────────────────────

def create_access_token(
    subject: str,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """
    Mint a signed JWT containing *subject* (the user-id) and optional
    extra claims.  Expires after ``jwt_access_token_expire_minutes``.
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    claims: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": expire,
    }
    if extra_claims:
        claims.update(extra_claims)

    return jwt.encode(
        claims,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify and decode a JWT. Raises ``JWTError`` on failure."""
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )


# ── FastAPI Dependency ───────────────────────────────────────────────────

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: str = Depends(_oauth2_scheme),
) -> dict[str, Any]:
    """
    FastAPI dependency that:
    1. Extracts the Bearer token from the ``Authorization`` header.
    2. Decodes the JWT.
    3. Loads the user from MongoDB.
    4. Returns the user document dict (with ``_id`` as a string).

    Raises ``401`` if any step fails.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await UserRepository.find_by_id(user_id)
    if user is None:
        raise credentials_exception

    return user
