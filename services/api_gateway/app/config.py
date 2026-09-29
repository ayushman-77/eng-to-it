"""
Centralised application settings loaded from environment variables.

All service connection strings, secrets, and tuning knobs are declared
here via pydantic-settings so they can be overridden at runtime without
touching source code.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Immutable application settings sourced from environment / .env file."""

    # ── MongoDB ──────────────────────────────────────────────────────────
    mongo_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI.",
    )
    mongo_db_name: str = Field(
        default="eng_to_it",
        description="Name of the primary Mongo database.",
    )

    # ── RabbitMQ ─────────────────────────────────────────────────────────
    rabbitmq_uri: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        description="AMQP connection URI for the RabbitMQ broker.",
    )
    rabbitmq_queue: str = Field(
        default="translation_tasks",
        description="Queue name that translation jobs are published to.",
    )
    rabbitmq_exchange: str = Field(
        default="translation_exchange",
        description="Direct exchange bound to the task queue.",
    )
    rabbitmq_routing_key: str = Field(
        default="translate",
        description="Routing key for translation task messages.",
    )

    # ── JWT Authentication ───────────────────────────────────────────────
    jwt_secret_key: str = Field(
        default="CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32",
        description="Secret key used to sign JWT access tokens.",
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(
        default=60,
        description="Access-token lifetime in minutes.",
    )

    # ── Server ───────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    debug: bool = Field(default=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance — import this throughout the app.
settings = Settings()
