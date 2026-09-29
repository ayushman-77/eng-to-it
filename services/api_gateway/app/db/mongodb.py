"""
Async MongoDB connection manager.

Uses Motor (async driver for PyMongo) to provide a singleton client
that is lazily initialised on first access and torn down on application
shutdown via the FastAPI lifespan hook.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings


class MongoDBManager:
    """Manages the lifecycle of the async MongoDB connection pool."""

    _client: AsyncIOMotorClient | None = None  # type: ignore[type-arg]
    _db: AsyncIOMotorDatabase | None = None  # type: ignore[type-arg]

    @classmethod
    async def connect(cls) -> None:
        """Open the connection pool and verify the server is reachable."""
        cls._client = AsyncIOMotorClient(
            settings.mongo_uri,
            maxPoolSize=50,
            minPoolSize=5,
            serverSelectionTimeoutMS=5000,
        )
        cls._db = cls._client[settings.mongo_db_name]

        # Verify connectivity on startup.
        await cls._client.admin.command("ping")

        # Ensure indexes exist for the collections we use.
        users = cls._db["users"]
        await users.create_index("username", unique=True)
        await users.create_index("email", unique=True)

        jobs = cls._db["translation_jobs"]
        await jobs.create_index("user_id")
        await jobs.create_index("status")
        await jobs.create_index([("user_id", 1), ("created_at", -1)])

    @classmethod
    async def disconnect(cls) -> None:
        """Gracefully close every socket in the pool."""
        if cls._client is not None:
            cls._client.close()
            cls._client = None
            cls._db = None

    @classmethod
    def get_db(cls) -> AsyncIOMotorDatabase:  # type: ignore[type-arg]
        """Return the database handle. Raises if not connected."""
        if cls._db is None:
            raise RuntimeError(
                "MongoDB is not connected. "
                "Call MongoDBManager.connect() during application startup."
            )
        return cls._db
