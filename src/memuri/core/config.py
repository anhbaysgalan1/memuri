"""Configuration management for the memuri SDK."""

import os
from typing import Dict, List, Literal, Optional, Union, Any

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    json_logs: bool = Field(False, description="Whether to output logs as JSON")


class TelemetrySettings(BaseSettings):
    """Telemetry and metrics configuration."""
    
    enabled: bool = Field(True, description="Whether to collect telemetry")
    prometheus_enabled: bool = Field(True, description="Whether to expose Prometheus metrics")
    metrics_port: int = Field(8000, description="Port to expose metrics on")
    anonymous_usage_stats: bool = Field(
        False, description="Whether to send anonymous usage statistics"
    )


class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    
    postgres_url: str = Field(
        ..., 
        description="PostgreSQL connection URL including pgvector extension"
    )
    pool_size: int = Field(10, description="Connection pool size")
    max_overflow: int = Field(20, description="Maximum connection overflow")
    pool_timeout: int = Field(30, description="Connection pool timeout in seconds")
    
    @validator("postgres_url")
    def validate_postgres_url(cls, v):
        """Validate that the PostgreSQL URL is properly formatted."""
        if not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("PostgreSQL URL must start with postgresql:// or postgres://")
        return v


class RedisSettings(BaseSettings):
    """Redis connection settings for cache and Celery broker."""
    
    redis_url: str = Field(..., description="Redis connection URL")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")
    cache_max_memory: str = Field("100mb", description="Maximum memory for Redis cache")
    
    @validator("redis_url")
    def validate_redis_url(cls, v):
        """Validate that the Redis URL is properly formatted."""
        if not v.startswith("redis://"):
            raise ValueError("Redis URL must start with redis://")
        return v


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""
    
    provider: Literal["pgvector", "milvus", "qdrant", "redis_vector"] = Field(
        "pgvector", description="Vector store provider"
    )
    dimension: int = Field(1536, description="Embedding dimension")
    dimensions: int = Field(1536, description="Embedding dimensions (alias for dimension)")
    index_type: str = Field("hnsw", description="Index type for vector store")
    connection_string: Optional[str] = Field(None, description="Connection string for the vector store")
    collection_name: Optional[str] = Field(None, description="Collection/table name for the vector store")
    
    # HNSW-specific settings
    ef_construction: int = Field(200, description="ef_construction parameter for HNSW index")
    ef_search: int = Field(100, description="ef_search parameter for HNSW index")
    m: int = Field(16, description="M parameter for HNSW index")


class EmbeddingSettings(BaseSettings):
    """Embedding service configuration."""
    
    provider: Literal["openai", "google", "sentence_transformers", "custom"] = Field(
        "openai", description="Embedding provider"
    )
    model_name: str = Field(
        "text-embedding-3-small", description="Embedding model name"
    )
    batch_size: int = Field(32, description="Batch size for embedding calls")
    max_tokens: int = Field(8191, description="Maximum tokens for embedding")
    
    # API keys and provider-specific configuration
    api_key: Optional[str] = Field(None, description="API key for the embedding provider")
    base_url: Optional[str] = Field(None, description="Base URL for the API (OpenAI, Ollama, etc.)")
    embedding_dims: Optional[int] = Field(None, description="Dimensions of the embedding model")
    http_client_proxies: Optional[Dict[str, str]] = Field(None, description="Proxy server settings")
    
    # Provider-specific settings
    azure_kwargs: Optional[Dict[str, Any]] = Field(None, description="Settings for Azure OpenAI")
    model_kwargs: Optional[Dict[str, Any]] = Field(None, description="Key-Value arguments for models")
    
    # Vertex AI specific settings
    credentials_json: Optional[str] = Field(
        None, description="Path to the Google Cloud credentials JSON file"
    )
    
    class Config:
        """Pydantic config for EmbeddingSettings."""
        extra = "allow"  # Allow extra fields for future compatibility


class LLMSettings(BaseSettings):
    """LLM service configuration."""
    
    provider: Literal["openai", "google", "custom"] = Field(
        "openai", description="LLM provider"
    )
    model_name: str = Field("gpt-3.5-turbo", description="LLM model name")
    temperature: float = Field(0.7, description="Temperature for LLM generation")
    max_tokens: int = Field(800, description="Maximum tokens for generation")
    
    # API keys and provider-specific configuration
    api_key: Optional[str] = Field(None, description="API key for the LLM provider")
    base_url: Optional[str] = Field(None, description="Base URL for the API")
    
    class Config:
        """Pydantic config for LLMSettings."""
        extra = "allow"  # Allow extra fields for future compatibility
    

class MemoryRuleSettings(BaseSettings):
    """Memory rule configuration for a specific category."""
    
    threshold: float = Field(0.5, description="Confidence threshold")
    action: Literal["add", "none", "short_term"] = Field(
        "add", description="Action to take when rule matches"
    )


class FeedbackSettings(BaseSettings):
    """Feedback service configuration."""
    
    retrain_interval: int = Field(3600, description="Retraining interval in seconds")
    min_samples_per_category: int = Field(
        10, description="Minimum samples per category for retraining"
    )
    
    class TrainingParams(BaseSettings):
        """Nested training parameters."""
        
        learning_rate: float = Field(0.01, description="Learning rate for training")
        batch_size: int = Field(32, description="Batch size for training")
    
    training_params: TrainingParams = Field(default_factory=TrainingParams)


class RerankSettings(BaseSettings):
    """Reranking configuration."""
    
    model_name: str = Field(
        "sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    alpha: float = Field(0.8, description="Weight for cross-encoder score")
    beta: float = Field(0.1, description="Weight for time decay")
    gamma: float = Field(0.1, description="Weight for metadata score")
    cache_results: bool = Field(True, description="Whether to cache reranking results")
    cache_ttl: int = Field(3600, description="TTL for reranking cache in seconds")


class MemuriSettings(BaseSettings):
    """Main configuration for the memuri SDK."""
    
    model_config = SettingsConfigDict(
        env_prefix="MEMURI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Core settings
    app_name: str = Field("memuri", description="Application name")
    debug: bool = Field(False, description="Debug mode")
    
    # Component settings
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    database: Optional[DatabaseSettings] = Field(None)
    redis: Optional[RedisSettings] = Field(None)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    
    # Memory settings
    memory_rules: Dict[str, MemoryRuleSettings] = Field(
        default_factory=lambda: {
            "TASK": MemoryRuleSettings(threshold=0.75, action="add"),
            "QUESTION": MemoryRuleSettings(action="none"),
            "EMOTION": MemoryRuleSettings(action="short_term"),
            "TODO": MemoryRuleSettings(action="add"),
        }
    )
    
    # Services configuration
    feedback: FeedbackSettings = Field(default_factory=FeedbackSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    
    # Auto memory settings
    auto_memory: bool = Field(True, description="Whether to enable automatic memory")
    hotwords: List[str] = Field(
        default_factory=lambda: ["remember", "note", "todo", "important"],
        description="Hotwords that trigger memory storage"
    )


def get_settings() -> MemuriSettings:
    """Get the application settings.
    
    This function loads settings from environment variables and .env file.
    
    Returns:
        MemuriSettings: The application settings
    """
    # Required settings must be provided via environment variables if not in .env
    database_settings = None
    redis_settings = None
    
    # Try to get database settings if environment variables are set
    if os.environ.get("MEMURI_DATABASE__POSTGRES_URL"):
        database_settings = DatabaseSettings(
            postgres_url=os.environ.get("MEMURI_DATABASE__POSTGRES_URL", "")
        )
    
    # Try to get Redis settings if environment variables are set
    if os.environ.get("MEMURI_REDIS__REDIS_URL"):
        redis_settings = RedisSettings(
            redis_url=os.environ.get("MEMURI_REDIS__REDIS_URL", "")
        )
    
    # Get embedding settings with API keys from environment
    embedding_settings = EmbeddingSettings(
        provider=os.environ.get("MEMURI_EMBEDDING__PROVIDER", "openai"),
        model_name=os.environ.get("MEMURI_EMBEDDING__MODEL_NAME", "text-embedding-3-small"),
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("MEMURI_EMBEDDING__API_KEY"),
    )
    
    # Get vector store settings from environment
    vector_store_settings = VectorStoreSettings(
        provider=os.environ.get("MEMURI_VECTOR_STORE__PROVIDER", "pgvector"),
        connection_string=os.environ.get("POSTGRES_CONNECTION") or os.environ.get("MEMURI_VECTOR_STORE__CONNECTION_STRING"),
        collection_name=os.environ.get("MEMURI_VECTOR_STORE__COLLECTION_NAME", "memories"),
    )
    
    # Get LLM settings with API keys from environment 
    llm_settings = LLMSettings(
        provider=os.environ.get("MEMURI_LLM__PROVIDER", "openai"),
        model_name=os.environ.get("MEMURI_LLM__MODEL_NAME", "gpt-3.5-turbo"),
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("MEMURI_LLM__API_KEY"),
    )
    
    # Create and return settings
    return MemuriSettings(
        debug=os.environ.get("MEMURI_DEBUG", "false").lower() == "true",
        database=database_settings,
        redis=redis_settings,
        embedding=embedding_settings,
        vector_store=vector_store_settings,
        llm=llm_settings,
    ) 