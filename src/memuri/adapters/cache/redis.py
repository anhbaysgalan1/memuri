"""
Redis adapter for caching operations.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic
import json
import time

import redis.asyncio as redis
import orjson

from memuri.domain.interfaces import CacheAdapter

T = TypeVar('T')


class RedisCache(CacheAdapter, Generic[T]):
    """Redis implementation of the Cache adapter."""

    def __init__(
        self,
        redis_url: str,
        namespace: str = "memuri",
        default_ttl: int = 3600,  # 1 hour
    ):
        """Initialize Redis cache adapter.
        
        Args:
            redis_url: Redis connection string
            namespace: Namespace for keys
            default_ttl: Default time-to-live in seconds
        """
        self.redis_url = redis_url
        self.namespace = namespace
        self.default_ttl = default_ttl
        self.client = None
        
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        self.client = redis.from_url(self.redis_url, decode_responses=True)
        
    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            
    def _make_key(self, key: str) -> str:
        """Create a namespaced key."""
        return f"{self.namespace}:{key}"
        
    async def get(self, key: str) -> Optional[T]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value or None if not found
        """
        if not self.client:
            await self.initialize()
            
        namespaced_key = self._make_key(key)
        raw_value = await self.client.get(namespaced_key)
        
        if raw_value is None:
            return None
            
        try:
            return orjson.loads(raw_value)
        except Exception:
            # Fallback to regular json if orjson fails
            return json.loads(raw_value)
            
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default TTL)
        """
        if not self.client:
            await self.initialize()
            
        namespaced_key = self._make_key(key)
        ttl = ttl if ttl is not None else self.default_ttl
        
        try:
            # Try using orjson first for better performance
            serialized = orjson.dumps(value)
        except Exception:
            # Fallback to regular json if orjson fails
            serialized = json.dumps(value)
            
        await self.client.set(namespaced_key, serialized, ex=ttl)
        
    async def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        if not self.client:
            await self.initialize()
            
        namespaced_key = self._make_key(key)
        await self.client.delete(namespaced_key)
        
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key exists, False otherwise
        """
        if not self.client:
            await self.initialize()
            
        namespaced_key = self._make_key(key)
        return bool(await self.client.exists(namespaced_key))
        
    async def expire(self, key: str, ttl: int) -> None:
        """Set a new expiration time for a key.
        
        Args:
            key: Cache key
            ttl: New time-to-live in seconds
        """
        if not self.client:
            await self.initialize()
            
        namespaced_key = self._make_key(key)
        await self.client.expire(namespaced_key, ttl)
        
    async def ttl(self, key: str) -> int:
        """Get the time-to-live for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Time-to-live in seconds or -2 if the key doesn't exist
        """
        if not self.client:
            await self.initialize()
            
        namespaced_key = self._make_key(key)
        return await self.client.ttl(namespaced_key)
        
    async def clear(self, pattern: str = "*") -> int:
        """Clear all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (default: "*" for all keys)
            
        Returns:
            Number of keys deleted
        """
        if not self.client:
            await self.initialize()
            
        namespaced_pattern = self._make_key(pattern)
        cursor = 0
        deleted_count = 0
        
        while True:
            cursor, keys = await self.client.scan(cursor, match=namespaced_pattern, count=100)
            if keys:
                deleted_count += await self.client.delete(*keys)
            if cursor == 0:
                break
                
        return deleted_count
        
    # Advanced caching methods for vector operations

    async def cache_search_results(
        self, 
        query_hash: str, 
        results: List[Tuple[Any, float]],
        ttl: Optional[int] = None,
    ) -> None:
        """Cache search results with scores.
        
        Args:
            query_hash: Hash of the query embedding
            results: List of (document, score) tuples
            ttl: Time-to-live in seconds
        """
        # We simplify the results to make them serializable
        simplified_results = []
        for doc, score in results:
            # Convert to dict and remove embedding to save space
            doc_dict = doc.dict() if hasattr(doc, "dict") else vars(doc)
            if "embedding" in doc_dict:
                del doc_dict["embedding"]
            simplified_results.append((doc_dict, score))
            
        await self.set(f"search:{query_hash}", simplified_results, ttl)
        
    async def get_search_results(
        self, 
        query_hash: str,
    ) -> Optional[List[Tuple[Dict[str, Any], float]]]:
        """Get cached search results.
        
        Args:
            query_hash: Hash of the query embedding
            
        Returns:
            List of (document dict, score) tuples or None
        """
        return await self.get(f"search:{query_hash}") 