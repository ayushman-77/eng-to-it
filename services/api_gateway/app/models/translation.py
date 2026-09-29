"""
Pydantic schemas for translation job request / response payloads and the
MongoDB document representation.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Finite state machine for a translation job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Request Schemas ──────────────────────────────────────────────────────

class TranslationRequest(BaseModel):
    """Payload accepted when a user submits text for translation."""
    source_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=["I am reading a book in my room."],
    )
    source_lang: str = Field(default="en", pattern=r"^[a-z]{2}$")
    target_lang: str = Field(default="it", pattern=r"^[a-z]{2}$")


# ── Response Schemas ─────────────────────────────────────────────────────

class TranslationJobResponse(BaseModel):
    """Returned immediately after a job is enqueued."""
    job_id: str
    status: JobStatus
    message: str = "Translation job enqueued successfully."


class TranslationResultResponse(BaseModel):
    """Full translation result — polled or retrieved from history."""
    job_id: str
    user_id: str
    source_text: str
    translated_text: Optional[str] = None
    source_lang: str
    target_lang: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[float] = None
    error_message: Optional[str] = None


class TranslationHistoryResponse(BaseModel):
    """Paginated translation history for a user."""
    user_id: str
    total_count: int
    page: int
    page_size: int
    translations: list[TranslationResultResponse]


# ── Internal / DB Document ──────────────────────────────────────────────

class TranslationJobDocument(BaseModel):
    """Shape of the translation job document stored in MongoDB."""
    user_id: str
    source_text: str
    translated_text: Optional[str] = None
    source_lang: str = "en"
    target_lang: str = "it"
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[float] = None
    error_message: Optional[str] = None
