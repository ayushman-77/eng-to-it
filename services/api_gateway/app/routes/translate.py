"""
Translation routes — submit a job, poll status, browse history.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth.security import get_current_user
from app.db.repositories import TranslationJobRepository, UserRepository
from app.models.translation import (
    TranslationRequest,
    TranslationJobResponse,
    TranslationResultResponse,
    TranslationHistoryResponse,
    TranslationJobDocument,
    JobStatus,
)
from app.queue.publisher import RabbitMQPublisher

router = APIRouter(prefix="/translate", tags=["Translation"])


# ── POST /translate ─────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=TranslationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a text for English → Italian translation",
)
async def submit_translation(
    payload: TranslationRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TranslationJobResponse:
    """
    Accepts source text, persists a **pending** job document in MongoDB,
    publishes the task payload to the RabbitMQ queue, and returns the
    job ID immediately (fire-and-forget, async processing).

    Requires a valid Bearer token.
    """
    user_id: str = current_user["_id"]

    # 1. Persist the pending job.
    job_doc = TranslationJobDocument(
        user_id=user_id,
        source_text=payload.source_text,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
    )
    job_id = await TranslationJobRepository.create(job_doc)

    # 2. Bump the user's translation counter.
    await UserRepository.increment_translation_count(user_id)

    # 3. Publish the task to RabbitMQ.
    await RabbitMQPublisher.publish_translation_task(
        job_id=job_id,
        user_id=user_id,
        source_text=payload.source_text,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
    )

    return TranslationJobResponse(job_id=job_id, status=JobStatus.PENDING)


# ── GET /translate/{job_id} ─────────────────────────────────────────────

@router.get(
    "/{job_id}",
    response_model=TranslationResultResponse,
    summary="Poll the status / result of a translation job",
)
async def get_translation_result(
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Retrieve the current state of a translation job.  The client can
    poll this endpoint until ``status`` transitions to ``completed``
    or ``failed``.
    """
    job = await TranslationJobRepository.find_by_id(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Translation job '{job_id}' not found.",
        )

    # Users may only view their own jobs.
    if job["user_id"] != current_user["_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this job.",
        )

    return job


# ── GET /translate/history/me ───────────────────────────────────────────

@router.get(
    "/history/me",
    response_model=TranslationHistoryResponse,
    summary="Retrieve paginated translation history for the current user",
)
async def get_my_translation_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TranslationHistoryResponse:
    """
    Returns the authenticated user's translation history, newest first,
    with simple offset-based pagination.
    """
    user_id: str = current_user["_id"]
    docs, total = await TranslationJobRepository.find_by_user(
        user_id, page=page, page_size=page_size
    )

    return TranslationHistoryResponse(
        user_id=user_id,
        total_count=total,
        page=page,
        page_size=page_size,
        translations=[TranslationResultResponse(**d) for d in docs],
    )
