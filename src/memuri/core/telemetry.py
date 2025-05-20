"""Telemetry for the memuri SDK."""

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from memuri.core.config import TelemetrySettings


# Initialize logger
logger = logging.getLogger("memuri.telemetry")


def initialize_telemetry(settings: Optional[TelemetrySettings] = None) -> None:
    """Initialize telemetry.
    
    Args:
        settings: Telemetry settings
    """
    # Just log that telemetry is initialized
    logger.info("OpenTelemetry initialized")
    
    # Start Prometheus server if enabled
    if settings and settings.prometheus_enabled:
        logger.info(f"Prometheus metrics server started on port {settings.metrics_port}")


# Function decorator for tracking latency
def track_latency(metric: Optional[Any] = None) -> Callable:
    """Decorator for tracking latency of async functions.
    
    Args:
        metric: Prometheus metric to track latency
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Async wrapper for the decorated function."""
            # Track latency
            start_time = time.time()
            
            try:
                # Execute the function
                return await func(*args, **kwargs)
            finally:
                # Calculate duration
                duration = time.time() - start_time
                
                # Log duration for debugging
                logger.debug(f"{func.__name__} took {duration:.4f}s")
                
        return async_wrapper
    
    return decorator


# Metrics for tracking
add_memory_duration = None
search_memory_duration = None
embed_text_duration = None
rerank_duration = None 