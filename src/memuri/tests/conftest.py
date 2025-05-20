"""
Pytest fixtures for testing.
"""
import asyncio
import os
import uuid
from typing import Dict, Any, List, AsyncGenerator

import pytest
import pytest_asyncio
import asyncpg
import redis.asyncio as redis
from pgvector.asyncpg import register_vector

from memuri.adapters.cache.redis import RedisCache
from memuri.adapters.db.client import PostgresClient
from memuri.adapters.vectorstore.pgvector import PGVectorAdapter
from memuri.adapters.vectorstore.redis_vector import RedisVectorAdapter
from memuri.domain.models import Document, DocumentChunk
from memuri.services.embedding import EmbeddingService
from memuri.services.memory import MemoryService
from memuri.services.retrieval import RetrievalService
from memuri.core.config import settings


# Test environment settings - override with environment variables or defaults
TEST_POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/test_memuri"
)
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379")


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def postgres_client() -> AsyncGenerator[PostgresClient, None]:
    """Initialize a PostgreSQL client for testing."""
    client = PostgresClient(TEST_POSTGRES_URL)
    await client.initialize()
    
    # Ensure tables exist with test data
    await client.execute("""
    DROP TABLE IF EXISTS memory_entries CASCADE;
    DROP TABLE IF EXISTS user_feedback CASCADE;
    
    CREATE TABLE memory_entries (
        id SERIAL PRIMARY KEY,
        memory_id TEXT NOT NULL UNIQUE,
        content TEXT NOT NULL,
        category TEXT,
        metadata JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    CREATE TABLE user_feedback (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        category TEXT NOT NULL,
        source TEXT DEFAULT 'user',
        used_for_training BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    CREATE INDEX memory_entries_category_idx ON memory_entries(category);
    CREATE INDEX memory_entries_created_at_idx ON memory_entries(created_at);
    CREATE INDEX user_feedback_category_idx ON user_feedback(category);
    CREATE INDEX user_feedback_used_for_training_idx ON user_feedback(used_for_training);
    """)
    
    yield client
    
    # Clean up tables
    await client.execute("""
    DROP TABLE IF EXISTS memory_entries CASCADE;
    DROP TABLE IF EXISTS user_feedback CASCADE;
    """)
    
    await client.close()


@pytest_asyncio.fixture(scope="session")
async def redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Initialize a Redis client for testing."""
    r = redis.from_url(TEST_REDIS_URL, decode_responses=True)
    await r.flushdb()  # Clear test database
    
    yield r
    
    await r.flushdb()
    await r.close()


@pytest_asyncio.fixture(scope="session")
async def redis_cache() -> AsyncGenerator[RedisCache, None]:
    """Initialize a Redis cache for testing."""
    cache = RedisCache(TEST_REDIS_URL, namespace="test_memuri", default_ttl=60)
    await cache.initialize()
    
    yield cache
    
    await cache.clear()
    await cache.close()


@pytest_asyncio.fixture(scope="session")
async def pgvector_adapter() -> AsyncGenerator[PGVectorAdapter, None]:
    """Initialize a pgvector adapter for testing."""
    # Create a separate table for testing
    test_table = f"test_vectors_{uuid.uuid4().hex[:8]}"
    adapter = PGVectorAdapter(
        connection_string=TEST_POSTGRES_URL,
        table_name=test_table,
        embedding_dim=4,  # Small dimension for testing
    )
    await adapter.initialize()
    
    yield adapter
    
    # Clean up table
    if adapter.pool:
        async with adapter.pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {test_table}")
    
    await adapter.close()


@pytest_asyncio.fixture(scope="session")
async def redis_vector_adapter() -> AsyncGenerator[RedisVectorAdapter, None]:
    """Initialize a Redis vector adapter for testing."""
    # Create a separate index for testing
    test_index = f"test_vectors_{uuid.uuid4().hex[:8]}"
    adapter = RedisVectorAdapter(
        redis_url=TEST_REDIS_URL,
        index_name=test_index,
        prefix=f"{test_index}:",
        embedding_dim=4,  # Small dimension for testing
    )
    await adapter.initialize()
    
    yield adapter
    
    # Clean up index
    if adapter.client:
        try:
            await adapter.client.ft(test_index).dropindex()
        except:
            pass  # Index might not exist
    
    await adapter.close()


@pytest.fixture
def sample_documents() -> List[Document]:
    """Generate sample documents for testing."""
    return [
        Document(
            id=f"doc_{i}",
            content=f"This is test document {i} about testing",
            embedding=[0.1, 0.2, 0.3, 0.4],  # 4D vector for testing
            metadata={"category": "TEST", "importance": i % 3}
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_document_chunks() -> List[DocumentChunk]:
    """Generate sample document chunks for testing."""
    chunks = []
    for i in range(3):
        for j in range(2):
            chunks.append(
                DocumentChunk(
                    document_id=f"doc_{i}",
                    id=f"chunk_{j}",
                    content=f"This is chunk {j} of document {i}",
                    embedding=[0.2, 0.3, 0.4, 0.5],  # 4D vector for testing
                    metadata={"category": "TEST", "chunk_index": j}
                )
            )
    return chunks


# Mock classes for testing
class MockEmbedder:
    """Mock embedder for testing."""
    
    async def embed(self, text: str) -> List[float]:
        """Return a fixed embedding for testing."""
        return [0.1, 0.2, 0.3, 0.4]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return fixed embeddings for testing."""
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    
    async def close(self) -> None:
        """Mock close method."""
        pass


@pytest_asyncio.fixture
async def mock_embedding_service() -> AsyncGenerator[EmbeddingService, None]:
    """Create a mock embedding service for testing."""
    service = EmbeddingService(embedder=MockEmbedder())
    
    yield service


@pytest_asyncio.fixture
async def retrieval_service(
    pgvector_adapter, redis_cache
) -> AsyncGenerator[RetrievalService, None]:
    """Create a retrieval service for testing."""
    service = RetrievalService(
        vector_store=pgvector_adapter,
        cache=redis_cache,
    )
    
    yield service


@pytest_asyncio.fixture
async def memory_service(
    mock_embedding_service, retrieval_service, redis_cache, postgres_client
) -> AsyncGenerator[MemoryService, None]:
    """Create a memory service for testing."""
    service = MemoryService(
        embedding_service=mock_embedding_service,
        retrieval_service=retrieval_service,
        cache=redis_cache,
        db=postgres_client,
    )
    
    yield service 