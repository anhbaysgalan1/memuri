"""Feedback service for classifier training."""

from typing import Dict, List, Optional, Any

from memuri.core.config import FeedbackSettings
from memuri.domain.models import MemoryCategory


class FeedbackService:
    """Service for collecting and processing feedback for the classifier."""
    
    def __init__(self, settings: Optional[FeedbackSettings] = None):
        """Initialize the feedback service.
        
        Args:
            settings: Optional feedback settings
        """
        self.settings = settings or FeedbackSettings()
        self.feedback_store: List[Dict[str, Any]] = []
    
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
        # Store feedback
        self.feedback_store.append({
            "text": text,
            "category": category,
            "metadata": metadata or {},
        })
    
    async def retrain_classifier(self) -> None:
        """Retrain the classifier using collected feedback."""
        # Check if we have enough feedback
        if len(self.feedback_store) < self.settings.min_samples_per_category:
            return
        
        # Get texts and categories from feedback
        texts = [feedback["text"] for feedback in self.feedback_store]
        categories = [feedback["category"] for feedback in self.feedback_store]
        
        # In a real implementation, we would get the classifier
        # and train it with the feedback data
        pass 