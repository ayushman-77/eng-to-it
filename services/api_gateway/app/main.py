"""
Distributed Translation Engine — API Gateway
=============================================

FastAPI application entry-point.

Lifecycle:
    startup  → connect to MongoDB & RabbitMQ
    shutdown → gracefully disconnect from both
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.mongodb import MongoDBManager
from app.queue.publisher import RabbitMQPublisher
from app.routes import auth, translate

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)


# ── Application Lifespan ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup / shutdown of external connections."""
    # Startup
    logger.info("Connecting to MongoDB at %s …", settings.mongo_uri)
    await MongoDBManager.connect()
    logger.info("MongoDB connected ✓")

    logger.info("Connecting to RabbitMQ at %s …", settings.rabbitmq_uri)
    await RabbitMQPublisher.connect()
    logger.info("RabbitMQ connected ✓")

    yield  # ← application is serving requests

    # Shutdown
    logger.info("Shutting down external connections …")
    await RabbitMQPublisher.disconnect()
    await MongoDBManager.disconnect()
    logger.info("All connections closed ✓")


# ── FastAPI Application ──────────────────────────────────────────────────

app = FastAPI(
    title="Distributed Translation Engine",
    description=(
        "Asynchronous English → Italian translation service.\n\n"
        "Translation requests are enqueued via RabbitMQ and processed "
        "by a fleet of Python PyTorch inference workers that update results "
        "directly in MongoDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount versioned routers.
API_V1 = "/api/v1"
app.include_router(auth.router, prefix=API_V1)
app.include_router(translate.router, prefix=API_V1)


# ── Health Check ─────────────────────────────────────────────────────────

@app.get("/health", tags=["Infrastructure"])
async def health_check():
    """Lightweight liveness probe."""
    return {"status": "healthy", "service": "api_gateway"}


# ── Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
