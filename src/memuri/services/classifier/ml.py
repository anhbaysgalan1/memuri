"""Machine learning-based classifier for memories."""

from typing import Dict, List, Optional

from memuri.core.categories import get_parent_category
from memuri.core.logging import get_logger
from memuri.domain.models import MemoryCategory

logger = get_logger(__name__)


class MLClassifier:
    """Machine learning classifier for memory categorization."""
    
    def __init__(self):
        """Initialize the ML classifier."""
        # This is a stub implementation that would normally initialize a model
        # In a real implementation, we would load a trained model here
        self._model = None
        logger.info("Initialized ML classifier (stub implementation)")
        
        # Fallback scores for when model is not trained
        self._default_categories = [
            MemoryCategory.PROFILE_INFORMATION,
            MemoryCategory.PREFERENCES,
            MemoryCategory.GOALS_ASPIRATIONS,
            MemoryCategory.PROJECTS_TASKS,
            MemoryCategory.EXPERIENCES_MEMORIES,
            MemoryCategory.MISCELLANEOUS
        ]
    
    async def classify(self, text: str) -> Dict[MemoryCategory, float]:
        """Classify text into memory categories.
        
        Args:
            text: Text to classify
            
        Returns:
            Dict[MemoryCategory, float]: Categories with confidence scores
        """
        # Initialize all categories with a small score
        scores: Dict[MemoryCategory, float] = {
            category: 0.01 for category in MemoryCategory
        }
        
        # If we had a real model, we would use it here
        if self._model is not None:
            # In a real implementation, this would be a model prediction
            pass
        else:
            # No model available, return a default distribution
            # Give higher scores to default categories
            for category in self._default_categories:
                scores[category] = 0.10
            
            # Give highest score to miscellaneous
            scores[MemoryCategory.MISCELLANEOUS] = 0.30
            
            # Apply parent-child relationships
            self._propagate_scores(scores)
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for category in scores:
                scores[category] /= total
        
        return scores
    
    def _propagate_scores(self, scores: Dict[MemoryCategory, float]) -> None:
        """Propagate scores from parents to children and vice versa.
        
        Args:
            scores: Dictionary of category scores to update in-place
        """
        # Copy scores to avoid modifying during iteration
        original_scores = scores.copy()
        
        # Propagate parent scores to children
        for category, score in original_scores.items():
            if score > 0.05:  # Only propagate significant scores
                category_str = category.value
                
                # Check if this is a parent category
                if category_str in [
                    "profile_information", "preferences", "goals_aspirations",
                    "routines_habits", "events_appointments", "projects_tasks",
                    "health_wellness", "social_relationships", "skills_knowledge",
                    "experiences_memories", "feedback_opinions", "financial_info",
                    "media_content", "contextual_metadata", "miscellaneous"
                ]:
                    # Find child categories
                    for child_category in MemoryCategory:
                        child_str = child_category.value
                        parent_str = get_parent_category(child_str)
                        
                        # If this child belongs to our parent
                        if parent_str == category_str and child_str != category_str:
                            # Boost child score
                            scores[child_category] += score * 0.3
        
        # Also propagate child scores to parents
        for category, score in original_scores.items():
            if score > 0.05:  # Only propagate significant scores
                category_str = category.value
                parent_str = get_parent_category(category_str)
                
                # If this is a child category
                if parent_str != category_str:
                    try:
                        parent_category = MemoryCategory(parent_str)
                        scores[parent_category] += score * 0.5
                    except ValueError:
                        pass  # Parent category not in enum
    
    async def train(self, texts: List[str], categories: List[MemoryCategory]) -> None:
        """Train the classifier on new examples.
        
        Args:
            texts: Training texts
            categories: Corresponding categories
        """
        # In a real implementation, we would train a model here
        # For now, just log that training was attempted
        category_counts = {}
        parent_category_counts = {}
        
        # Count occurrences of each category
        for category in categories:
            category_str = category.value
            category_counts[category_str] = category_counts.get(category_str, 0) + 1
            
            # Also count parent categories
            parent_str = get_parent_category(category_str)
            if parent_str != category_str:
                parent_category_counts[parent_str] = parent_category_counts.get(parent_str, 0) + 1
        
        logger.info(f"Would train classifier with {len(texts)} examples")
        logger.info(f"Category distribution: {category_counts}")
        logger.info(f"Parent category distribution: {parent_category_counts}")
        
        # In a real implementation, this would update self._model 