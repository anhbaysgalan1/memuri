"""
Celery application configuration.
"""
from celery import Celery
from celery.schedules import crontab

from memuri.core.config import settings

# Configure Celery
redis_url = settings.redis_url
app = Celery(
    "memuri", 
    broker=f"{redis_url}/0",  # Use database 0 as broker
    backend=f"{redis_url}/1",  # Use database 1 as result backend
)

# Configure serialization
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]

# Configure task routes
app.conf.task_routes = {
    "memuri.workers.embed_tasks.*": {"queue": "embedding"},
    "memuri.workers.memory_tasks.*": {"queue": "memory"},
}

# Configure periodic tasks
app.conf.beat_schedule = {
    # Clean up memories daily at 3 AM
    "cleanup-old-memories": {
        "task": "memuri.workers.memory_tasks.cleanup_memories",
        "schedule": crontab(hour=3, minute=0),  # Run at 3:00 AM
        "args": (90, 10000, ["IMPORTANT", "DECISION"]),  # Keep important memories
    },
    
    # Update feedback classifier every 4 hours
    "update-feedback-classifier": {
        "task": "memuri.workers.memory_tasks.update_feedback_classifier",
        "schedule": crontab(minute=0, hour="*/4"),  # Run every 4 hours
    },
}

# Set prefetch multiplier (how many tasks a worker prefetches)
app.conf.worker_prefetch_multiplier = 4

# Set concurrency (default number of worker processes)
app.conf.worker_concurrency = 4

# Set task time limit (in seconds)
app.conf.task_time_limit = 300  # 5 minutes

# Set soft time limit (in seconds)
app.conf.task_soft_time_limit = 240  # 4 minutes

# Set task rate limits
app.conf.task_annotations = {
    "memuri.workers.embed_tasks.embed_batch": {"rate_limit": "100/m"},
    "memuri.workers.embed_tasks.embed_document": {"rate_limit": "100/m"},
}

# Configure Celery logging
app.conf.worker_hijack_root_logger = False 