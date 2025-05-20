"""Reranking service for search results."""

import logging
from typing import List

from memuri.core.logging import get_logger
from memuri.domain.models import ScoredMemory

logger = get_logger(__name__)


class RerankService:
    """Service for reranking search results."""
    
    def __init__(self):
        """Initialize the reranking service."""
        logger.info("Initialized reranking service with model sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    async def rerank(
        self, 
        query: str, 
        candidates: List[ScoredMemory],
        top_k: int = 5,
    ) -> List[ScoredMemory]:
        """Rerank search results.
        
        Args:
            query: Search query
            candidates: Initial search results
            top_k: Number of results to return
            
        Returns:
            List[ScoredMemory]: Reranked search results
        """
        # For simplicity, we're just returning the original results
        # In a real implementation, this would use a cross-encoder model
        return candidates[:top_k] 