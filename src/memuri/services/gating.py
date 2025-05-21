"""Memory gating service for filtering memories before storage."""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from memuri.core.config import MemuriSettings, get_settings
from memuri.core.logging import get_logger
from memuri.domain.interfaces import ClassifierService, EmbeddingService, MemoryService
from memuri.domain.models import Memory, MemoryCategory, MemorySource

logger = get_logger(__name__)


class MemoryGate:
    """Gate that decides whether memories should be stored.
    
    This sits between the agent/API layer and the memory store, 
    filtering out irrelevant information before storage.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        memory_service: Optional[MemoryService] = None,
        classifier_service: Optional[ClassifierService] = None,
        settings: Optional[MemuriSettings] = None,
        similarity_threshold: float = 0.85,
        confidence_threshold: float = 0.4,
        min_content_length: int = 30,
        skip_words: Optional[List[str]] = None,
        keep_phrases: Optional[List[str]] = None,
        max_recent_embeddings: int = 100,
    ):
        """Initialize the memory gate.
        
        Args:
            embedding_service: Service for creating embeddings
            memory_service: Optional service for checking similarity to recent memories
            classifier_service: Optional service for classifying relevance
            settings: Optional settings
            similarity_threshold: Threshold for similarity check (higher = more strict filtering)
            confidence_threshold: Threshold for classifier confidence (higher = more strict filtering)
            min_content_length: Minimum content length to store
            skip_words: List of words/phrases that signal content should be skipped
            keep_phrases: List of words/phrases that signal content should be kept
            max_recent_embeddings: Maximum number of recent embeddings to keep in memory
        """
        self.embedding_service = embedding_service
        self.memory_service = memory_service
        self.classifier_service = classifier_service
        self.settings = settings or get_settings()
        
        # Gating parameters
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.min_content_length = min_content_length
        self.skip_words = skip_words or ["ok", "thanks", "thank you", "sure", "alright", "got it"]
        self.keep_phrases = keep_phrases or ["remember", "note", "don't forget", "important"]
        
        # In-memory cache of recent embeddings for quick similarity checks
        self.recent_embeddings: List[List[float]] = []
        self.max_recent_embeddings = max_recent_embeddings
        
        logger.info("Initialized memory gate")
    
    async def evaluate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Evaluate whether text should be stored in memory.
        
        Args:
            text: Text to evaluate
            metadata: Optional metadata that might inform the decision
            
        Returns:
            Tuple[bool, str]: Decision (True=keep, False=skip) and reason
        """
        # Apply rule-based heuristics first (fast and cheap)
        if not self._passes_basic_rules(text):
            return False, "Failed basic rules check"
        
        # If we have explicit keep phrases, prioritize those
        if self._has_keep_phrases(text):
            return True, "Contains explicit keep phrase"
        
        # Check similarity to recent memories (if memory service available)
        if self.recent_embeddings:
            similar, similarity = await self._check_similarity(text)
            if similar:
                return False, f"Similar to recent memory (score: {similarity:.2f})"
        
        # Run classifier if available (most expensive check)
        if self.classifier_service:
            relevant, category, score = await self._classify_relevance(text)
            if not relevant:
                return False, f"Low relevance score ({score:.2f})"
            return True, f"Classified as {category.value} with score {score:.2f}"
            
        # Default to storing if no classifier available
        return True, "Passed all checks"
    
    async def evaluate_and_store(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[MemoryCategory] = None,
        source: MemorySource = MemorySource.USER,
    ) -> Tuple[bool, str, Optional[Memory]]:
        """Evaluate whether text should be stored and store it if it passes.
        
        Args:
            text: Text to evaluate and potentially store
            metadata: Optional metadata
            category: Optional memory category
            source: Source of the memory
            
        Returns:
            Tuple[bool, str, Optional[Memory]]: 
                Decision (True=kept, False=skipped),
                Reason for the decision,
                Memory object if stored, None otherwise
        """
        # Skip storage if no memory service
        if not self.memory_service:
            logger.warning("No memory service available for storage")
            return False, "No memory service available", None
        
        # Evaluate if should store
        should_store, reason = await self.evaluate(text, metadata)
        
        if not should_store:
            return False, reason, None
            
        # Determine category if not provided
        if not category and self.classifier_service:
            _, category, _ = await self._classify_relevance(text)
        
        # Default category if still not determined
        category = category or MemoryCategory.GENERAL
        
        # Create embedding
        embedding_response = await self.embedding_service.embed_texts([text])
        embedding = embedding_response.embeddings[0]
        
        # Add to recent embeddings cache
        self._add_to_recent_embeddings(embedding)
        
        # Create memory object
        memory = Memory(
            content=text,
            category=category,
            source=source,
            embedding=embedding,
            metadata=metadata or {},
        )
        
        # Store the memory
        stored_memory = await self.memory_service.add(memory)
        
        return True, reason, stored_memory
    
    def _passes_basic_rules(self, text: str) -> bool:
        """Apply basic rule-based heuristics.
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if passes basic rules, False otherwise
        """
        # Check minimum length
        if len(text.strip()) < self.min_content_length:
            logger.debug(f"Text too short: {len(text.strip())} chars")
            return False
            
        # Check for skip words/phrases (exact matches)
        text_lower = text.lower()
        for word in self.skip_words:
            if word.lower() == text_lower:
                logger.debug(f"Text matches skip word: {word}")
                return False
                
        return True
    
    def _has_keep_phrases(self, text: str) -> bool:
        """Check if text contains any phrases that signal it should be kept.
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if contains keep phrases, False otherwise
        """
        text_lower = text.lower()
        for phrase in self.keep_phrases:
            if phrase.lower() in text_lower:
                logger.debug(f"Text contains keep phrase: {phrase}")
                return True
                
        return False
    
    async def _check_similarity(self, text: str) -> Tuple[bool, float]:
        """Check if text is similar to recent memories.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple[bool, float]: (is_similar, similarity_score)
        """
        if not self.recent_embeddings:
            return False, 0.0
            
        # Get embedding for text
        embedding_response = await self.embedding_service.embed_texts([text])
        query_embedding = embedding_response.embeddings[0]
        
        # Calculate similarity with recent embeddings
        max_similarity = 0.0
        for existing_embedding in self.recent_embeddings:
            similarity = self._cosine_similarity(query_embedding, existing_embedding)
            max_similarity = max(max_similarity, similarity)
            
            if similarity > self.similarity_threshold:
                logger.debug(f"Text similar to recent memory: {similarity:.4f}")
                return True, similarity
                
        return False, max_similarity
    
    async def _classify_relevance(self, text: str) -> Tuple[bool, MemoryCategory, float]:
        """Classify text relevance.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple[bool, MemoryCategory, float]: 
                (is_relevant, best_category, confidence_score)
        """
        if not self.classifier_service:
            return True, MemoryCategory.GENERAL, 1.0
            
        # Classify text
        scores = await self.classifier_service.classify(text)
        
        # Find highest scoring category
        best_category = MemoryCategory.GENERAL
        best_score = 0.0
        
        for category, score in scores.items():
            if score > best_score:
                best_score = score
                best_category = category
                
        # Check if confidence exceeds threshold
        is_relevant = best_score >= self.confidence_threshold
        
        return is_relevant, best_category, best_score
    
    def _add_to_recent_embeddings(self, embedding: List[float]) -> None:
        """Add embedding to recent embeddings cache.
        
        Args:
            embedding: Embedding to add
        """
        self.recent_embeddings.append(embedding)
        
        # Trim if needed
        if len(self.recent_embeddings) > self.max_recent_embeddings:
            self.recent_embeddings.pop(0)  # Remove oldest
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            float: Cosine similarity
        """
        a_array = np.array(a)
        b_array = np.array(b)
        
        dot_product = np.dot(a_array, b_array)
        norm_a = np.linalg.norm(a_array)
        norm_b = np.linalg.norm(b_array)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b) 