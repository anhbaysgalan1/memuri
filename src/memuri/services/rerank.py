"""Reranking service for improving search results."""

import math
import time
from typing import Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

from memuri.core.config import RerankSettings, get_settings
from memuri.core.logging import get_logger
from memuri.core.telemetry import rerank_duration, track_latency
from memuri.domain.interfaces import RerankingService
from memuri.domain.models import ScoredMemory

logger = get_logger(__name__)


class RerankService(RerankingService):
    """Service for reranking search results using a cross-encoder model."""
    
    def __init__(self, settings: Optional[RerankSettings] = None):
        """Initialize the reranking service.
        
        Args:
            settings: Optional reranking settings
        """
        self.settings = settings or get_settings().rerank
        
        # Initialize cross-encoder
        self.model = None  # Lazy-loaded on first use
        
        # Reranking parameters
        self.alpha = self.settings.alpha  # Weight for cross-encoder score
        self.beta = self.settings.beta    # Weight for time decay
        self.gamma = self.settings.gamma  # Weight for metadata score
        
        logger.info(f"Initialized reranking service with model {self.settings.model_name}")
    
    def _ensure_model_loaded(self):
        """Ensure the cross-encoder model is loaded."""
        if self.model is None:
            self.model = CrossEncoder(self.settings.model_name, max_length=512)
            logger.info(f"Loaded cross-encoder model {self.settings.model_name}")
    
    def _calculate_time_decay(self, age_seconds: float) -> float:
        """Calculate time decay score.
        
        This function calculates a decay factor based on the age of the memory.
        Newer memories get a higher score.
        
        Args:
            age_seconds: Age of the memory in seconds
            
        Returns:
            float: Time decay score between 0.0 and 1.0
        """
        # Convert to days for more meaningful decay
        age_days = age_seconds / (24 * 3600)
        
        # Exponential decay with half-life of 30 days
        half_life = 30
        decay = math.exp(-age_days * math.log(2) / half_life)
        
        return decay
    
    def _calculate_metadata_score(self, memory: ScoredMemory) -> float:
        """Calculate metadata score.
        
        This function calculates a score based on metadata. It can be customized
        to consider various metadata factors.
        
        Args:
            memory: Memory to score
            
        Returns:
            float: Metadata score between 0.0 and 1.0
        """
        # Example: boost score for pinned memories
        if memory.memory.metadata.get("pinned", False):
            return 1.0
            
        # Example: boost DECISION category
        if memory.memory.category.value == "DECISION":
            return 0.8
            
        # Example: boost memories with high importance
        importance = memory.memory.metadata.get("importance", 0.5)
        
        return min(1.0, importance)
    
    @track_latency(rerank_duration)
    async def rerank(
        self, 
        query: str, 
        candidates: List[ScoredMemory],
        top_k: int = 5,
    ) -> List[ScoredMemory]:
        """Rerank search results using a cross-encoder model.
        
        Args:
            query: Search query
            candidates: Initial search results
            top_k: Number of results to return
            
        Returns:
            List[ScoredMemory]: Reranked search results
        """
        if not candidates:
            return []
            
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Prepare inputs for cross-encoder
        # Each input is a tuple (query, candidate_text)
        cross_encoder_inputs = [(query, memory.memory.content) for memory in candidates]
        
        # Use the model to score the (query, candidate) pairs
        # Note: CrossEncoder is not async, so we need to run it in a thread pool
        import asyncio
        cross_scores = await asyncio.to_thread(self.model.predict, cross_encoder_inputs)
        
        # Calculate time decay and metadata scores
        time_decay_scores = [
            self._calculate_time_decay(memory.memory.created_at.timestamp()) 
            for memory in candidates
        ]
        
        metadata_scores = [
            self._calculate_metadata_score(memory)
            for memory in candidates
        ]
        
        # Combine scores
        # final_score = α * cross_score + β * time_decay + γ * metadata_score
        combined_scores = [
            self.alpha * cross_score + 
            self.beta * time_decay +
            self.gamma * metadata
            for cross_score, time_decay, metadata in zip(
                cross_scores, time_decay_scores, metadata_scores
            )
        ]
        
        # Create new scored memories with combined scores
        reranked_memories = [
            ScoredMemory(memory=memory.memory, score=score)
            for memory, score in zip(candidates, combined_scores)
        ]
        
        # Sort by score and take top_k
        reranked_memories.sort(key=lambda x: x.score, reverse=True)
        
        return reranked_memories[:top_k] 