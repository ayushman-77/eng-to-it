"""
Data-access repositories for users and translation jobs.

Each repository encapsulates all MongoDB queries for its domain entity,
keeping route handlers free of raw database logic.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId

from app.db.mongodb import MongoDBManager
from app.models.user import UserDocument
from app.models.translation import TranslationJobDocument, JobStatus


# ── User Repository ─────────────────────────────────────────────────────

class UserRepository:
    """CRUD operations on the `users` collection."""

    @staticmethod
    def _collection():
        return MongoDBManager.get_db()["users"]

    @classmethod
    async def create(cls, user: UserDocument) -> str:
        """Insert a new user document and return its string ObjectId."""
        doc = user.model_dump()
        result = await cls._collection().insert_one(doc)
        return str(result.inserted_id)

    @classmethod
    async def find_by_username(cls, username: str) -> Optional[dict[str, Any]]:
        """Look up a user by unique username."""
        doc = await cls._collection().find_one({"username": username})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    @classmethod
    async def find_by_email(cls, email: str) -> Optional[dict[str, Any]]:
        """Look up a user by unique email."""
        doc = await cls._collection().find_one({"email": email})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    @classmethod
    async def find_by_id(cls, user_id: str) -> Optional[dict[str, Any]]:
        """Look up a user by ObjectId string."""
        doc = await cls._collection().find_one({"_id": ObjectId(user_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    @classmethod
    async def increment_translation_count(cls, user_id: str) -> None:
        """Atomically bump the user's translation counter by one."""
        await cls._collection().update_one(
            {"_id": ObjectId(user_id)},
            {"$inc": {"translation_count": 1}},
        )


# ── Translation Job Repository ──────────────────────────────────────────

class TranslationJobRepository:
    """CRUD operations on the `translation_jobs` collection."""

    @staticmethod
    def _collection():
        return MongoDBManager.get_db()["translation_jobs"]

    @classmethod
    async def create(cls, job: TranslationJobDocument) -> str:
        """Insert a pending translation job and return its string ObjectId."""
        doc = job.model_dump()
        result = await cls._collection().insert_one(doc)
        return str(result.inserted_id)

    @classmethod
    async def find_by_id(cls, job_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a single job by ObjectId string."""
        doc = await cls._collection().find_one({"_id": ObjectId(job_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
            doc["job_id"] = doc.pop("_id")
        return doc

    @classmethod
    async def update_status(
        cls,
        job_id: str,
        status: JobStatus,
        translated_text: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Transition a job to a new status, optionally attaching results."""
        update_fields: dict[str, Any] = {"status": status.value}
        if translated_text is not None:
            update_fields["translated_text"] = translated_text
        if processing_time_ms is not None:
            update_fields["processing_time_ms"] = processing_time_ms
        if error_message is not None:
            update_fields["error_message"] = error_message
        if status in (JobStatus.COMPLETED, JobStatus.FAILED):
            update_fields["completed_at"] = datetime.now(timezone.utc)

        await cls._collection().update_one(
            {"_id": ObjectId(job_id)},
            {"$set": update_fields},
        )

    @classmethod
    async def find_by_user(
        cls,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Return a page of jobs belonging to *user_id* (newest first)
        together with the total count.
        """
        filt = {"user_id": user_id}
        total = await cls._collection().count_documents(filt)

        cursor = (
            cls._collection()
            .find(filt)
            .sort("created_at", -1)
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        docs: list[dict[str, Any]] = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            doc["job_id"] = doc.pop("_id")
            docs.append(doc)

        return docs, total
