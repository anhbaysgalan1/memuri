"""Feedback service for handling classifier feedback and retraining."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import aioredis

from memuri.core.config import FeedbackSettings, get_settings
from memuri.core.logging import get_logger
from memuri.domain.interfaces import ClassifierService, FeedbackService as FeedbackServiceInterface
from memuri.domain.models import MemoryCategory

logger = get_logger(__name__)


class FeedbackService(FeedbackServiceInterface):
    """Service for handling classifier feedback and orchestrating retraining."""
    
    def __init__(
        self,
        classifier_service: Optional[ClassifierService] = None,
        redis_client: Optional[aioredis.Redis] = None,
        settings: Optional[FeedbackSettings] = None,
    ):
        """Initialize the feedback service.
        
        Args:
            classifier_service: Classifier service to retrain
            redis_client: Redis client for storing feedback
            settings: Feedback settings
        """
        self.classifier_service = classifier_service
        self.redis = redis_client
        self.settings = settings or get_settings().feedback
        
        # Redis keys
        self.feedback_key = "memuri:feedback"
        self.last_retrain_key = "memuri:feedback:last_retrain"
        
        logger.info("Initialized feedback service")
    
    async def _ensure_redis(self) -> aioredis.Redis:
        """Ensure Redis client is available.
        
        Returns:
            aioredis.Redis: Redis client
            
        Raises:
            RuntimeError: If Redis is not available
        """
        if self.redis:
            return self.redis
        
        try:
            # Create Redis client
            self.redis = await aioredis.from_url(
                get_settings().redis.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            return self.redis
        except Exception as e:
            logger.error(f"Failed to connect to Redis for feedback: {e}")
            raise RuntimeError(f"Failed to connect to Redis: {e}")
    
    async def log_feedback(
        self, 
        text: str, 
        category: MemoryCategory,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log feedback for the classifier.
        
        Args:
            text: Text that was classified
            category: Correct category
            metadata: Optional metadata
            
        Raises:
            RuntimeError: If Redis is not available
        """
        try:
            redis = await self._ensure_redis()
            
            # Create feedback entry
            feedback_entry = {
                "text": text,
                "category": category.value,
                "metadata": metadata or {},
                "timestamp": time.time(),
            }
            
            # Add to Redis list
            await redis.lpush(
                self.feedback_key,
                json.dumps(feedback_entry)
            )
            
            logger.debug(f"Logged feedback for category {category.value}")
            
            # Check if we should retrain
            await self._check_retrain()
            
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
            raise
    
    async def get_feedback(
        self, 
        limit: Optional[int] = None,
        category: Optional[MemoryCategory] = None,
    ) -> List[Dict[str, Any]]:
        """Get feedback entries.
        
        Args:
            limit: Maximum number of entries to return
            category: Optional category filter
            
        Returns:
            List[Dict[str, Any]]: Feedback entries
            
        Raises:
            RuntimeError: If Redis is not available
        """
        try:
            redis = await self._ensure_redis()
            
            # Get all feedback entries
            all_entries = await redis.lrange(self.feedback_key, 0, -1)
            
            # Parse JSON entries
            entries = [json.loads(entry) for entry in all_entries]
            
            # Filter by category if provided
            if category:
                entries = [
                    entry for entry in entries 
                    if entry["category"] == category.value
                ]
            
            # Apply limit if provided
            if limit is not None:
                entries = entries[:limit]
                
            return entries
            
        except Exception as e:
            logger.error(f"Failed to get feedback: {e}")
            raise
    
    async def _check_retrain(self) -> None:
        """Check if retraining is due and initiate if needed."""
        if not self.classifier_service:
            logger.warning("No classifier service provided, cannot retrain")
            return
            
        try:
            redis = await self._ensure_redis()
            
            # Get last retrain timestamp
            last_retrain = await redis.get(self.last_retrain_key)
            last_retrain_time = float(last_retrain) if last_retrain else 0
            
            # Get current time
            current_time = time.time()
            
            # Check if retraining is due
            if current_time - last_retrain_time > self.settings.retrain_interval:
                # Check if we have enough samples
                feedback = await self.get_feedback()
                
                # Count samples per category
                category_counts = {}
                for entry in feedback:
                    category = entry["category"]
                    category_counts[category] = category_counts.get(category, 0) + 1
                
                # Check if any category has enough samples
                has_enough_samples = any(
                    count >= self.settings.min_samples_per_category
                    for count in category_counts.values()
                )
                
                if has_enough_samples:
                    # Schedule retraining
                    asyncio.create_task(self.retrain_classifier())
                else:
                    logger.debug(
                        f"Not enough samples for retraining, need {self.settings.min_samples_per_category} per category"
                    )
            
        except Exception as e:
            logger.error(f"Error checking for retraining: {e}")
    
    async def retrain_classifier(self) -> None:
        """Retrain the classifier using collected feedback.
        
        Raises:
            RuntimeError: If Redis or classifier service is not available
        """
        if not self.classifier_service:
            logger.warning("No classifier service provided, cannot retrain")
            return
            
        try:
            redis = await self._ensure_redis()
            
            # Get all feedback entries
            feedback = await self.get_feedback()
            
            if not feedback:
                logger.info("No feedback entries for retraining")
                return
                
            # Extract texts and categories
            texts = [entry["text"] for entry in feedback]
            categories = [MemoryCategory(entry["category"]) for entry in feedback]
            
            # Log retraining
            logger.info(f"Retraining classifier with {len(texts)} examples")
            
            # Retrain the classifier
            await self.classifier_service.train(texts, categories)
            
            # Update last retrain timestamp
            await redis.set(self.last_retrain_key, time.time())
            
            logger.info("Classifier retraining completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to retrain classifier: {e}")
            raise 