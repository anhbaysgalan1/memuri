"""Retrieval service for vector similarity search with caching."""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import aioredis

from memuri.core.config import MemuriSettings, get_settings
from memuri.core.logging import get_logger
from memuri.core.telemetry import cache_hit_counter, cache_miss_counter, memory_ops_duration, track_latency
from memuri.domain.interfaces import EmbeddingService, MemoryService
from memuri.domain.models import MemoryCategory, ScoredMemory

logger = get_logger(__name__)


class RetrievalService:
    """Service for vector similarity search with caching."""
    
    def __init__(
        self, 
        memory_service: MemoryService,
        embedding_service: EmbeddingService,
        redis_client: Optional[aioredis.Redis] = None,
        settings: Optional[MemuriSettings] = None,
    ):
        """Initialize the retrieval service.
        
        Args:
            memory_service: Memory service for storage and retrieval
            embedding_service: Embedding service for text embeddings
            redis_client: Redis client for caching
            settings: Application settings
        """
        self.memory_service = memory_service
        self.embedding_service = embedding_service
        self.redis = redis_client
        self.settings = settings or get_settings()
        
        # Cache settings from the config
        self.cache_ttl = self.settings.redis.cache_ttl
        
        logger.info("Initialized retrieval service")
    
    def _generate_cache_key(
        self, 
        query: str,
        category: Optional[MemoryCategory] = None,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a cache key for the query.
        
        Args:
            query: Query string
            category: Optional category filter
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
            
        Returns:
            str: Cache key
        """
        # Create a dictionary of all parameters
        key_dict = {
            "query": query,
            "category": category.value if category else None,
            "top_k": top_k,
            "metadata_filters": metadata_filters or {},
        }
        
        # Convert to a stable JSON string
        key_str = json.dumps(key_dict, sort_keys=True)
        
        # Hash it
        return f"memuri:retrieval:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[List[ScoredMemory]]:
        """Get results from cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Optional[List[ScoredMemory]]: Cached results if found, None otherwise
        """
        if not self.redis:
            return None
        
        try:
            cached = await self.redis.get(cache_key)
            if cached:
                # Increment hit counter
                cache_hit_counter.labels(cache_type="retrieval").inc()
                
                # Deserialize and return
                memories_data = json.loads(cached)
                return [
                    ScoredMemory.model_validate(memory_data)
                    for memory_data in memories_data
                ]
            else:
                # Increment miss counter
                cache_miss_counter.labels(cache_type="retrieval").inc()
                return None
                
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def _store_in_cache(self, cache_key: str, results: List[ScoredMemory]) -> None:
        """Store results in cache.
        
        Args:
            cache_key: Cache key
            results: Results to cache
        """
        if not self.redis:
            return
        
        try:
            # Serialize results
            results_json = json.dumps([
                memory.model_dump()
                for memory in results
            ])
            
            # Store in Redis with TTL
            await self.redis.set(cache_key, results_json, ex=self.cache_ttl)
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
    
    @track_latency(memory_ops_duration, {"operation": "vector_search"})
    async def search_by_text(
        self,
        query: str,
        category: Optional[MemoryCategory] = None,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> List[ScoredMemory]:
        """Search for memories by text.
        
        This method embeds the query text and then searches by vector.
        
        Args:
            query: Query text
            category: Optional category filter
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
            use_cache: Whether to use the cache
            
        Returns:
            List[ScoredMemory]: Scored memories
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            query, category, top_k, metadata_filters
        )
        
        # Try to get from cache if enabled
        if use_cache:
            cached_results = await self._get_from_cache(cache_key)
            if cached_results:
                logger.debug(f"Cache hit for query: {query}")
                return cached_results
        
        # Embed the query
        embedding_response = await self.embedding_service.embed_texts([query])
        query_embedding = embedding_response.embeddings[0]
        
        # Search by vector
        results = await self.memory_service.search_by_vector(
            vector=query_embedding,
            category=category,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        
        # Cache results if enabled
        if use_cache:
            await self._store_in_cache(cache_key, results)
        
        return results
    
    @track_latency(memory_ops_duration, {"operation": "vector_search"})
    async def search_by_vector(
        self,
        vector: List[float],
        category: Optional[MemoryCategory] = None,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredMemory]:
        """Search for memories by vector.
        
        Args:
            vector: Query vector
            category: Optional category filter
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
            
        Returns:
            List[ScoredMemory]: Scored memories
        """
        # This method doesn't use caching as vectors are typically unique
        return await self.memory_service.search_by_vector(
            vector=vector,
            category=category,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        
    async def get_redis_client(self) -> Optional[aioredis.Redis]:
        """Get or create a Redis client.
        
        Returns:
            Optional[aioredis.Redis]: Redis client if available
        """
        if self.redis:
            return self.redis
        
        try:
            # Create Redis client
            self.redis = await aioredis.from_url(
                self.settings.redis.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            return self.redis
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return None 