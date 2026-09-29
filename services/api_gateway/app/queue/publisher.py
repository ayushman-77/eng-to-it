"""
Async RabbitMQ publisher.

Publishes translation task messages to a durable direct exchange so that
one or more C++ inference workers can compete-consume them.

Message payload (JSON):
    {
        "job_id":      "<ObjectId hex string>",
        "user_id":     "<ObjectId hex string>",
        "source_text": "I am reading a book in my room.",
        "source_lang": "en",
        "target_lang": "it"
    }
"""

import json
import logging
from typing import Any

import aio_pika

from app.config import settings

logger = logging.getLogger(__name__)


class RabbitMQPublisher:
    """Manages a persistent AMQP connection and publishes task messages."""

    _connection: aio_pika.abc.AbstractRobustConnection | None = None
    _channel: aio_pika.abc.AbstractChannel | None = None
    _exchange: aio_pika.abc.AbstractExchange | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    @classmethod
    async def connect(cls) -> None:
        """Open a robust connection, declare the exchange and queue, and bind them."""
        cls._connection = await aio_pika.connect_robust(settings.rabbitmq_uri)
        cls._channel = await cls._connection.channel()

        # Declare a durable direct exchange.
        cls._exchange = await cls._channel.declare_exchange(
            settings.rabbitmq_exchange,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )

        # Declare the durable task queue.
        queue = await cls._channel.declare_queue(
            settings.rabbitmq_queue,
            durable=True,
        )

        # Bind the queue to the exchange with the routing key.
        await queue.bind(cls._exchange, routing_key=settings.rabbitmq_routing_key)

        logger.info(
            "RabbitMQ publisher connected — exchange=%s, queue=%s",
            settings.rabbitmq_exchange,
            settings.rabbitmq_queue,
        )

    @classmethod
    async def disconnect(cls) -> None:
        """Close channel and connection."""
        if cls._channel and not cls._channel.is_closed:
            await cls._channel.close()
        if cls._connection and not cls._connection.is_closed:
            await cls._connection.close()
        cls._channel = None
        cls._connection = None
        cls._exchange = None
        logger.info("RabbitMQ publisher disconnected.")

    # ── Publishing ───────────────────────────────────────────────────────

    @classmethod
    async def publish_translation_task(
        cls,
        job_id: str,
        user_id: str,
        source_text: str,
        source_lang: str = "en",
        target_lang: str = "it",
    ) -> None:
        """
        Serialize and publish a single translation task.

        The message is marked **persistent** (delivery_mode=2) so it
        survives a broker restart.
        """
        if cls._exchange is None:
            raise RuntimeError(
                "RabbitMQ publisher is not connected. "
                "Call RabbitMQPublisher.connect() during application startup."
            )

        payload: dict[str, Any] = {
            "job_id": job_id,
            "user_id": user_id,
            "source_text": source_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

        message = aio_pika.Message(
            body=json.dumps(payload).encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json",
        )

        await cls._exchange.publish(
            message,
            routing_key=settings.rabbitmq_routing_key,
        )

        logger.info("Published translation task — job_id=%s", job_id)
