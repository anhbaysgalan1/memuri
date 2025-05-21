"""Feedback service for classifier training."""

from typing import Dict, List, Optional, Any

from memuri.core.categories import get_parent_category
from memuri.core.config import FeedbackSettings
from memuri.core.logging import get_logger
from memuri.domain.models import MemoryCategory

logger = get_logger(__name__)


class FeedbackService:
    """Service for collecting and processing feedback for the classifier."""
    
    def __init__(self, settings: Optional[FeedbackSettings] = None):
        """Initialize the feedback service.
        
        Args:
            settings: Optional feedback settings
        """
        self.settings = settings or FeedbackSettings()
        self.feedback_store: List[Dict[str, Any]] = []
        self.category_counts: Dict[str, int] = {}
        self.subcategory_counts: Dict[str, int] = {}
    
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
        """
        # Get the category value
        category_value = category.value
        
        # Get the parent category if this is a subcategory
        parent_category = get_parent_category(category_value)
        
        # Update counts
        if parent_category == category_value:
            # This is a main category
            self.category_counts[category_value] = self.category_counts.get(category_value, 0) + 1
        else:
            # This is a subcategory
            self.subcategory_counts[category_value] = self.subcategory_counts.get(category_value, 0) + 1
            self.category_counts[parent_category] = self.category_counts.get(parent_category, 0) + 1
            
        # Store feedback with parent category information
        self.feedback_store.append({
            "text": text,
            "category": category_value,
            "parent_category": parent_category,
            "metadata": metadata or {},
            "used_for_training": False,
        })
        
        logger.debug(f"Added feedback for text classified as {category_value} (parent: {parent_category})")
        
        # Check if we have enough samples to trigger retraining
        await self._check_retrain_threshold()
    
    async def _check_retrain_threshold(self) -> None:
        """Check if we have enough samples to retrain the classifier."""
        # Check minimum number of categories that have sufficient samples
        min_samples = self.settings.min_samples_per_category
        categories_with_samples = sum(1 for count in self.category_counts.values() if count >= min_samples)
        
        # If we have at least 3 categories with enough samples, retrain
        if categories_with_samples >= 3:
            await self.retrain_classifier()
    
    async def retrain_classifier(self) -> None:
        """Retrain the classifier using collected feedback."""
        # Check if we have any feedback
        if not self.feedback_store:
            logger.debug("No feedback available for retraining")
            return
        
        # Get unused feedback
        unused_feedback = [f for f in self.feedback_store if not f.get("used_for_training", False)]
        
        if not unused_feedback:
            logger.debug("No new feedback available for retraining")
            return
            
        logger.info(f"Retraining classifier with {len(unused_feedback)} new feedback samples")
        
        # In a real implementation, we would get the classifier
        # and train it with the feedback data
        # For now, just mark the feedback as used
        for feedback in self.feedback_store:
            feedback["used_for_training"] = True
            
        # Log category distributions
        logger.info(f"Category distribution: {self.category_counts}")
        logger.info(f"Subcategory distribution: {self.subcategory_counts}")
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get the counts of feedback by category.
        
        Returns:
            Dict[str, int]: Counts by category
        """
        return self.category_counts.copy()
    
    def get_subcategory_counts(self) -> Dict[str, int]:
        """Get the counts of feedback by subcategory.
        
        Returns:
            Dict[str, int]: Counts by subcategory
        """
        return self.subcategory_counts.copy() 