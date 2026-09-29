import asyncio
import json
import logging
import os
import os
from datetime import datetime, timezone

import aio_pika
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# Import the actual Transformer translation logic
from translate import load_inference_components, translate
from config import get_config

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "eng_to_it")
MONGO_JOBS_COLLECTION = "translation_jobs"

RABBITMQ_URI = os.getenv("RABBITMQ_URI", "amqp://guest:guest@localhost:5672/")
RABBITMQ_QUEUE = "translation_tasks"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def main():
    # 1. Load the ML Model and Inference Components
    logger.info("Loading PyTorch Transformer model... this might take a few seconds.")
    cfg, model, tokenizer_src, tokenizer_tgt, device = load_inference_components()
    logger.info("Transformer model loaded and ready!")

    # 2. Connect to MongoDB
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    db = mongo_client[MONGO_DB_NAME]
    coll = db[MONGO_JOBS_COLLECTION]
    logger.info(f"Connected to MongoDB: {MONGO_DB_NAME}.{MONGO_JOBS_COLLECTION}")

    # 2. Connect to RabbitMQ
    connection = await aio_pika.connect_robust(RABBITMQ_URI)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    
    queue = await channel.declare_queue(RABBITMQ_QUEUE, durable=True)
    logger.info(f"Connected to RabbitMQ. Listening on queue: {RABBITMQ_QUEUE}...")

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process(): # Auto-acks on success
                try:
                    payload = json.loads(message.body.decode())
                    job_id = payload["job_id"]
                    source_text = payload["source_text"]
                    source_lang = payload.get("source_lang", "en")
                    target_lang = payload.get("target_lang", "it")
                    
                    logger.info(f"Picked up job {job_id} | Text: {source_text[:30]}...")
                    
                    # Update status to processing
                    await coll.update_one(
                        {"_id": ObjectId(job_id)},
                        {"$set": {"status": "processing"}}
                    )
                    
                    # ──────────────────────────────────────────────
                    # Run actual Transformer Inference
                    # ──────────────────────────────────────────────
                    logger.info(f"Running ML inference for: {source_text[:50]}...")
                    
                    import time
                    start_time = time.time()
                    try:
                        translated_text = translate(
                            sentence=source_text,
                            config=cfg,
                            model=model,
                            tokenizer_src=tokenizer_src,
                            tokenizer_tgt=tokenizer_tgt,
                            device=device
                        )
                    except Exception as inf_err:
                        logger.error(f"Inference error: {inf_err}")
                        translated_text = f"[Inference Error] {inf_err}"
                    
                    processing_time_ms = (time.time() - start_time) * 1000.0
                    
                    # Update status to completed
                    await coll.update_one(
                        {"_id": ObjectId(job_id)},
                        {"$set": {
                            "status": "completed",
                            "translated_text": translated_text,
                            "processing_time_ms": processing_time_ms,
                            "completed_at": datetime.now(timezone.utc)
                        }}
                    )
                    
                    logger.info(f"Completed job {job_id} successfully.")
                except Exception as e:
                    logger.error(f"Failed to process message: {e}")
                    if "job_id" in locals():
                        await coll.update_one(
                            {"_id": ObjectId(job_id)},
                            {"$set": {
                                "status": "failed",
                                "error_message": str(e),
                                "completed_at": datetime.now(timezone.utc)
                            }}
                        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped.")
