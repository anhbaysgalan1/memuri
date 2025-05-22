"""
Celery worker configuration for Memuri.

This module sets up the Celery application for background tasks processing.
"""

from celery import Celery
import os

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Create Celery instance
app = Celery('memuri')

# Load configuration from environment variables with CELERY_ prefix
app.conf.update(
    broker_url=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    result_backend=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes
    task_soft_time_limit=300,  # 5 minutes
    worker_max_tasks_per_child=200,
    worker_concurrency=4,
)

# Include task modules
app.autodiscover_tasks(['memuri.workers'], force=True)

# Optional task routes configuration
app.conf.task_routes = {
    'memuri.workers.embed_tasks.*': {'queue': 'embed'},
    'memuri.workers.memory_tasks.*': {'queue': 'memory'},
}

# Optional task queues configuration
app.conf.task_queues = {
    'embed': {
        'exchange': 'embed',
        'routing_key': 'embed',
    },
    'memory': {
        'exchange': 'memory',
        'routing_key': 'memory',
    },
}


@app.task(bind=True)
def debug_task(self):
    """Debug task to test Celery worker."""
    logger.info(f"Request: {self.request!r}")
    return "Debug task executed successfully"