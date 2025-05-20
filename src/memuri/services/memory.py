"""Memory service for orchestrating memory operations."""

import time
import uuid
from typing import Any, Dict, List, Optional, Union

from memuri.core.config import MemuriSettings
from memuri.core.logging import get_logger
from memuri.core.telemetry import memory_ops_duration, track_latency
from memuri.domain.interfaces import (
    ClassifierService,
    EmbeddingService,
    FeedbackService,
    MemoryService,
    RerankingService,
)
from memuri.domain.models import (
    Memory,
    MemoryCategory,
    MemorySource,
    SearchQuery,
    SearchResult,
    ScoredMemory,
)

logger = get_logger(__name__)


class MemoryOrchestrator:
    """Orchestrates memory operations across services."""
    
    def __init__(
        self,
        memory_service: MemoryService,
        embedding_service: EmbeddingService,
        reranking_service: RerankingService,
        classifier_service: ClassifierService,
        feedback_service: FeedbackService,
        settings: MemuriSettings,
    ):
        """Initialize the memory orchestrator.
        
        Args:
            memory_service: Memory service for storage and retrieval
            embedding_service: Embedding service for text embeddings
            reranking_service: Reranking service for search results
            classifier_service: Classifier service for categorizing content
            feedback_service: Feedback service for tracking and retraining
            settings: Application settings
        """
        self.memory_service = memory_service
        self.embedding_service = embedding_service
        self.reranking_service = reranking_service
        self.classifier_service = classifier_service
        self.feedback_service = feedback_service
        self.settings = settings
        
        logger.info("Initialized memory orchestrator")
    
    @track_latency(memory_ops_duration, {"operation": "add_memory"})
    async def add_memory(
        self,
        content: str,
        category: Optional[MemoryCategory] = None,
        source: MemorySource = MemorySource.USER,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Add a memory to the store.
        
        This method handles the end-to-end flow of adding a memory, including:
        1. Classifying the content if no category is provided
        2. Embedding the content
        3. Storing the memory in the memory service
        4. Optionally logging feedback for user-provided categories
        
        Args:
            content: Content of the memory
            category: Optional category, will be auto-classified if not provided
            source: Source of the memory
            metadata: Optional metadata to associate with the memory
            
        Returns:
            Memory: The added memory
        """
        # Classify content if no category is provided
        if category is None:
            # Get category predictions
            category_scores = await self.classifier_service.classify(content)
            
            # Find the highest scoring category
            if category_scores:
                top_category, top_score = max(
                    category_scores.items(), key=lambda x: x[1]
                )
                
                # Check if the score meets the threshold
                rule = self.settings.memory_rules.get(
                    top_category.value, 
                    self.settings.memory_rules.get("GENERAL")
                )
                
                if rule and top_score >= rule.threshold:
                    category = top_category
                    logger.debug(
                        f"Auto-classified content as {category.value} with score {top_score:.2f}"
                    )
                else:
                    # Default to GENERAL if no category meets the threshold
                    category = MemoryCategory.GENERAL
                    logger.debug(
                        f"No category met threshold, using {category.value} (best was {top_category.value} with {top_score:.2f})"
                    )
            else:
                # Default to GENERAL if classification failed
                category = MemoryCategory.GENERAL
                logger.warning("Classification failed, using GENERAL category")
        
        # Get the rule for this category
        rule = self.settings.memory_rules.get(
            category.value, 
            self.settings.memory_rules.get("GENERAL")
        )
        
        # Check if we should add this memory based on the rule
        if rule and rule.action == "none":
            logger.debug(f"Skipping memory add due to rule for {category.value}")
            # Return a memory object but don't actually store it
            return Memory(
                id=str(uuid.uuid4()),
                content=content,
                category=category,
                source=source,
                metadata=metadata or {},
            )
        
        # Embed the content
        embedding_response = await self.embedding_service.embed_texts([content])
        embedding = embedding_response.embeddings[0] if embedding_response.embeddings else None
        
        # Create the memory object
        memory = Memory(
            content=content,
            category=category,
            source=source,
            embedding=embedding,
            metadata=metadata or {},
        )
        
        # Store the memory
        if rule and rule.action == "short_term":
            # TODO: Handle short-term memory (in-memory cache only)
            logger.debug(f"Storing memory in short-term only for {category.value}")
            # For now, just store it normally
            stored_memory = await self.memory_service.add(memory)
        else:
            # Add to long-term memory
            stored_memory = await self.memory_service.add(memory)
        
        # If this is a user-provided category, log feedback
        if source == MemorySource.USER and category != MemoryCategory.GENERAL:
            await self.feedback_service.log_feedback(
                text=content,
                category=category,
                metadata={"memory_id": stored_memory.id},
            )
        
        return stored_memory
    
    @track_latency(memory_ops_duration, {"operation": "search_memory"})
    async def search_memory(self, query: SearchQuery) -> SearchResult:
        """Search for memories.
        
        This method handles the end-to-end flow of searching for memories, including:
        1. Searching the memory service
        2. Optionally reranking the results
        
        Args:
            query: Search query
            
        Returns:
            SearchResult: Search results
        """
        start_time = time.time()
        
        # Search the memory service
        search_result = await self.memory_service.search(query)
        
        # Rerank results if requested
        if query.rerank and search_result.memories:
            reranked_memories = await self.reranking_service.rerank(
                query=query.query,
                candidates=search_result.memories,
                top_k=query.top_k,
            )
            
            # Update the results
            search_result.memories = reranked_memories
            search_result.reranked = True
        
        # Calculate search time
        search_result.search_time = time.time() - start_time
        
        return search_result
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID.
        
        Args:
            memory_id: ID of the memory to get
            
        Returns:
            Optional[Memory]: Memory if found, None otherwise
        """
        return await self.memory_service.get(memory_id)
    
    async def batch_add_memories(self, memories: List[Memory]) -> List[Memory]:
        """Add a batch of memories to the store.
        
        Args:
            memories: Memuries to add
            
        Returns:
            List[Memory]: The added memories
        """
        # TODO: Implement batching logic for embeddings
        # This is a placeholder implementation that processes one at a time
        return await self.memory_service.add_batch(memories) 