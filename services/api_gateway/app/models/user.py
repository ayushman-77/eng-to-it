"""
Pydantic schemas for user-related request / response payloads and the
MongoDB document representation.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# ── Request Schemas ──────────────────────────────────────────────────────

class UserCreateRequest(BaseModel):
    """Payload accepted when registering a new user."""
    username: str = Field(..., min_length=3, max_length=64, examples=["ayushman"])
    email: EmailStr = Field(..., examples=["ayush@example.com"])
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(default=None, max_length=128)


class UserLoginRequest(BaseModel):
    """Payload accepted when authenticating."""
    username: str
    password: str


# ── Response Schemas ─────────────────────────────────────────────────────

class UserProfileResponse(BaseModel):
    """Public-facing user profile (never exposes the password hash)."""
    id: str = Field(..., alias="_id")
    username: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime
    translation_count: int = 0

    class Config:
        populate_by_name = True


class TokenResponse(BaseModel):
    """JWT token returned after successful authentication."""
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


# ── Internal / DB Document ──────────────────────────────────────────────

class UserDocument(BaseModel):
    """Shape of the user document stored in MongoDB."""
    username: str
    email: str
    full_name: Optional[str] = None
    hashed_password: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    translation_count: int = 0
