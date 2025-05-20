"""Keyword-based classifier for memories."""

from typing import Dict, List

from memuri.domain.models import MemoryCategory


class KeywordClassifier:
    """Simple keyword-based classifier for memory categorization."""
    
    def __init__(self):
        """Initialize the keyword classifier with default keywords."""
        # Define keywords for each category
        self.category_keywords = {
            MemoryCategory.PERSONAL: ["I", "me", "my", "mine", "personal", "myself", "family", "friend"],
            MemoryCategory.TASK: ["task", "do", "complete", "finish", "work", "project", "assignment"],
            MemoryCategory.QUESTION: ["who", "what", "where", "when", "why", "how", "question", "?"],
            MemoryCategory.EMOTION: ["feel", "happy", "sad", "angry", "frustrated", "excited", "emotion"],
            MemoryCategory.DECISION: ["decide", "choice", "option", "alternative", "select", "choose", "decision"],
            MemoryCategory.TODO: ["todo", "need to", "must", "should", "have to", "reminder", "don't forget"],
            MemoryCategory.GENERAL: [],  # Fallback category
        }
    
    async def classify(self, text: str) -> Dict[MemoryCategory, float]:
        """Classify text into memory categories.
        
        Args:
            text: Text to classify
            
        Returns:
            Dict[MemoryCategory, float]: Categories with confidence scores
        """
        # Convert text to lowercase for case-insensitive matching
        text = text.lower()
        
        # Initialize scores
        scores: Dict[MemoryCategory, float] = {
            category: 0.0 for category in MemoryCategory
        }
        
        # Set a minimum score for GENERAL
        scores[MemoryCategory.GENERAL] = 0.2
        
        # Count keyword occurrences for each category
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    scores[category] += 0.2  # Increase score for each keyword found
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for category in scores:
                scores[category] /= total
        
        return scores
    
    async def train(self, texts: List[str], categories: List[MemoryCategory]) -> None:
        """Train the classifier on new examples.
        
        This is a no-op for the keyword classifier.
        
        Args:
            texts: Training texts
            categories: Corresponding categories
        """
        # Keyword classifier doesn't learn, so this is a no-op
        pass 