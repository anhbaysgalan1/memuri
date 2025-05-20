"""Telemetry and metrics for the memuri SDK."""

import asyncio
import platform
import socket
import time
from typing import Any, Callable, Dict, Optional, TypeVar

import prometheus_client
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from memuri.core.config import TelemetrySettings, get_settings
from memuri.core.logging import get_logger

logger = get_logger(__name__)

# Define metric objects
embedding_latency = prometheus_client.Histogram(
    "memuri_embedding_latency_seconds",
    "Embedding request latency in seconds",
    ["provider", "model", "batch_size"]
)

cache_hit_counter = prometheus_client.Counter(
    "memuri_cache_hits_total",
    "Number of cache hits",
    ["cache_type"]
)

cache_miss_counter = prometheus_client.Counter(
    "memuri_cache_misses_total",
    "Number of cache misses",
    ["cache_type"]
)

memory_ops_duration = prometheus_client.Histogram(
    "memuri_memory_ops_duration_seconds",
    "Memory operation duration in seconds",
    ["operation", "store_type"]
)

vector_index_size = prometheus_client.Gauge(
    "memuri_vector_index_size",
    "Number of vectors in the index",
    ["store_type"]
)

llm_request_duration = prometheus_client.Histogram(
    "memuri_llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["provider", "model"]
)

rerank_duration = prometheus_client.Histogram(
    "memuri_rerank_duration_seconds",
    "Reranking operation duration in seconds",
    ["model"]
)

async_tasks_gauge = prometheus_client.Gauge(
    "memuri_async_tasks",
    "Number of active async tasks"
)


# Type variable for the return type of the decorated function
T = TypeVar("T")


def track_latency(
    metric: prometheus_client.Histogram, 
    labels: Optional[Dict[str, str]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to track latency of a function using a Prometheus histogram.
    
    Args:
        metric: Prometheus histogram to record latency in
        labels: Labels to apply to the metric
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """Decorator that tracks function execution time.
        
        Args:
            func: Function to track
            
        Returns:
            Callable: Wrapped function
        """
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            """Async wrapper for tracking latency.
            
            Args:
                *args: Function arguments
                **kwargs: Function keyword arguments
                
            Returns:
                Any: Function return value
            """
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            """Sync wrapper for tracking latency.
            
            Args:
                *args: Function arguments
                **kwargs: Function keyword arguments
                
            Returns:
                Any: Function return value
            """
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        
        # Return the appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def setup_prometheus_server(port: int = 8000) -> None:
    """Set up a Prometheus metrics server.
    
    Args:
        port: Port to expose metrics on
    """
    try:
        prometheus_client.start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start Prometheus server: {e}")


def setup_opentelemetry() -> None:
    """Set up OpenTelemetry for tracing and metrics."""
    # Create a resource with service info
    resource = Resource.create({
        "service.name": "memuri",
        "service.version": __import__("memuri").__version__,
        "deployment.environment": get_settings().app_name,
    })
    
    # Set up tracing
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    
    # Add a span processor that prints to the console
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    tracer_provider.add_span_processor(span_processor)
    
    # Set up metrics
    meter_provider = MeterProvider(resource=resource)
    metrics.set_meter_provider(meter_provider)
    
    logger.info("OpenTelemetry initialized")


def initialize_telemetry(settings: Optional[TelemetrySettings] = None) -> None:
    """Initialize telemetry collection for the SDK.
    
    Args:
        settings: Telemetry settings to use
    """
    if settings is None:
        settings = get_settings().telemetry
    
    if not settings.enabled:
        logger.info("Telemetry is disabled")
        return
    
    # Set up OpenTelemetry
    setup_opentelemetry()
    
    # Set up Prometheus server if enabled
    if settings.prometheus_enabled:
        setup_prometheus_server(settings.metrics_port)
    
    # Send anonymous usage ping if enabled
    if settings.anonymous_usage_stats:
        send_anonymous_usage_ping()


def send_anonymous_usage_ping() -> None:
    """Send anonymous usage statistics.
    
    This function collects anonymous usage data and sends it to our servers
    to help improve the SDK. No personal information is collected or sent.
    """
    try:
        # Gather system info
        system_info = {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
        }
        
        # In a real implementation, this would send the data to a server
        # For now, we just log it
        logger.info(f"Would send anonymous usage ping: {system_info}")
    except Exception as e:
        logger.error(f"Failed to send anonymous usage ping: {e}")


# Get a tracer for this module
tracer = trace.get_tracer(__name__) 