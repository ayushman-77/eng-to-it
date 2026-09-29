"""
Authentication routes — user registration and login.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.auth.security import hash_password, verify_password, create_access_token
from app.config import settings
from app.db.repositories import UserRepository
from app.models.user import (
    UserCreateRequest,
    UserLoginRequest,
    UserDocument,
    UserProfileResponse,
    TokenResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── POST /auth/register ─────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user account",
)
async def register_user(payload: UserCreateRequest) -> dict[str, Any]:
    """
    Register a new user.

    * Validates uniqueness of both *username* and *email*.
    * Stores a bcrypt-hashed password — never the plaintext.
    * Returns the created user profile.
    """
    # Guard: duplicate username.
    if await UserRepository.find_by_username(payload.username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{payload.username}' is already taken.",
        )

    # Guard: duplicate email.
    if await UserRepository.find_by_email(payload.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{payload.email}' is already registered.",
        )

    # Persist.
    user_doc = UserDocument(
        username=payload.username,
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
    )
    user_id = await UserRepository.create(user_doc)

    # Re-fetch to get the canonical representation with _id.
    created = await UserRepository.find_by_id(user_id)
    if created is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User was created but could not be retrieved.",
        )
    return created


# ── POST /auth/login ────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and receive a JWT access token",
)
async def login_user(payload: UserLoginRequest) -> TokenResponse:
    """
    Authenticate a user by *username* + *password* and return a signed
    JWT access token.
    """
    user = await UserRepository.find_by_username(payload.username)
    if user is None or not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        subject=user["_id"],
        extra_claims={"username": user["username"]},
    )

    return TokenResponse(
        access_token=token,
        expires_in_minutes=settings.jwt_access_token_expire_minutes,
    )
