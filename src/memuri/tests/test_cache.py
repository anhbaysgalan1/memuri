"""
Tests for the Redis cache adapter.
"""
import asyncio
import time
from typing import List, Dict, Any

import pytest
import redis.asyncio as redis

from memuri.adapters.cache.redis import RedisCache


@pytest.mark.asyncio
async def test_redis_cache_initialization():
    """Test that the Redis cache can be initialized."""
    cache = RedisCache("redis://localhost:6379/0", "test_init", 60)
    await cache.initialize()
    
    assert cache.client is not None
    
    await cache.close()


@pytest.mark.asyncio
async def test_redis_cache_set_get(redis_cache: RedisCache):
    """Test setting and getting values from Redis cache."""
    # Set a simple value
    await redis_cache.set("test_key", "test_value")
    value = await redis_cache.get("test_key")
    assert value == "test_value"
    
    # Set a complex value
    complex_value = {
        "name": "Test Object",
        "values": [1, 2, 3],
        "nested": {"a": 1, "b": 2}
    }
    await redis_cache.set("complex_key", complex_value)
    result = await redis_cache.get("complex_key")
    assert result == complex_value
    

@pytest.mark.asyncio
async def test_redis_cache_ttl(redis_cache: RedisCache):
    """Test TTL functionality of Redis cache."""
    # Set with default TTL
    await redis_cache.set("ttl_key", "ttl_value")
    ttl = await redis_cache.ttl("ttl_key")
    assert 0 < ttl <= 60  # Default TTL is 60 seconds
    
    # Set with custom TTL
    await redis_cache.set("custom_ttl", "value", 10)
    ttl = await redis_cache.ttl("custom_ttl")
    assert 0 < ttl <= 10
    
    # Set a new expiry time
    await redis_cache.expire("custom_ttl", 30)
    ttl = await redis_cache.ttl("custom_ttl")
    assert 10 < ttl <= 30
    

@pytest.mark.asyncio
async def test_redis_cache_delete(redis_cache: RedisCache):
    """Test deleting keys from Redis cache."""
    # Set a value
    await redis_cache.set("delete_key", "value")
    assert await redis_cache.exists("delete_key")
    
    # Delete it
    await redis_cache.delete("delete_key")
    assert not await redis_cache.exists("delete_key")
    
    # Delete non-existent key (should not error)
    await redis_cache.delete("non_existent")
    

@pytest.mark.asyncio
async def test_redis_cache_clear(redis_cache: RedisCache):
    """Test clearing keys from Redis cache."""
    # Set multiple values with different patterns
    await redis_cache.set("clear_test_1", "value1")
    await redis_cache.set("clear_test_2", "value2")
    await redis_cache.set("other_key", "value3")
    
    # Clear specific pattern
    count = await redis_cache.clear("clear_test_*")
    assert count == 2
    assert not await redis_cache.exists("clear_test_1")
    assert not await redis_cache.exists("clear_test_2")
    assert await redis_cache.exists("other_key")
    
    # Clear all
    await redis_cache.set("another_key", "value4")
    count = await redis_cache.clear()
    assert count >= 2  # Should at least delete the two keys we know exist


@pytest.mark.asyncio
async def test_redis_cache_search_results(redis_cache: RedisCache):
    """Test caching search results with scores."""
    # Create mock search results
    mock_results = [
        ({
            "id": "doc1",
            "content": "Test document 1",
            "metadata": {"category": "TEST"}
        }, 0.95),
        ({
            "id": "doc2",
            "content": "Test document 2",
            "metadata": {"category": "TEST"}
        }, 0.85),
    ]
    
    # Cache the results
    query_hash = "test_query_hash"
    await redis_cache.cache_search_results(query_hash, mock_results)
    
    # Retrieve the results
    cached_results = await redis_cache.get_search_results(query_hash)
    
    # Verify the results
    assert len(cached_results) == 2
    assert cached_results[0][0]["id"] == "doc1"
    assert cached_results[0][1] == 0.95
    assert cached_results[1][0]["id"] == "doc2"
    assert cached_results[1][1] == 0.85 