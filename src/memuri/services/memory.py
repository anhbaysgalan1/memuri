"""Memory orchestration services."""

import time
from typing import Dict, List, Optional, Any, Tuple

from memuri.core.config import MemuriSettings
from memuri.core.logging import get_logger
from memuri.core.telemetry import track_latency
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
)
from memuri.services.gating import MemoryGate

logger = get_logger(__name__)


class MemoryOrchestrator:
    """Memory orchestration service.
    
    This class orchestrates the interactions between different services involved
    in memory management, like embedding, storage, classification, etc.
    """
    
    def __init__(
        self,
        memory_service: MemoryService,
        embedding_service: EmbeddingService,
        reranking_service: RerankingService,
        classifier_service: ClassifierService,
        feedback_service: FeedbackService,
        memory_gate: Optional[MemoryGate] = None,
        settings: Optional[MemuriSettings] = None,
    ):
        """Initialize the memory orchestrator.
        
        Args:
            memory_service: Memory storage service
            embedding_service: Embedding service
            reranking_service: Reranking service
            classifier_service: Classifier service
            feedback_service: Feedback service
            memory_gate: Optional memory gating service
            settings: Optional settings
        """
        self.memory_service = memory_service
        self.embedding_service = embedding_service
        self.reranking_service = reranking_service
        self.classifier_service = classifier_service
        self.feedback_service = feedback_service
        self.settings = settings
        
        # Initialize memory gate if not provided
        self.memory_gate = memory_gate or MemoryGate(
            embedding_service=embedding_service,
            memory_service=memory_service,
            classifier_service=classifier_service,
            settings=settings,
        )
        
        logger.info("Initialized memory orchestrator")
        
    @track_latency()
    async def add_memory(
        self, 
        content: str, 
        category: Optional[MemoryCategory] = None,
        source: MemorySource = MemorySource.USER,
        metadata: Optional[Dict[str, Any]] = None,
        skip_gating: bool = False,
    ) -> Tuple[bool, str, Optional[Memory]]:
        """Add a memory to the store with gating.
        
        Args:
            content: Memory content
            category: Memory category
            source: Memory source
            metadata: Memory metadata
            skip_gating: Whether to skip the gating process
            
        Returns:
            Tuple[bool, str, Optional[Memory]]:
                - Whether the memory was stored
                - Reason for the decision
                - Memory object if stored, None otherwise
        """
        # Apply memory gating unless skipped
        if not skip_gating:
            stored, reason, memory = await self.memory_gate.evaluate_and_store(
                text=content,
                metadata=metadata,
                category=category,
                source=source,
            )
            
            if not stored:
                logger.info(f"Memory gating prevented storage: {reason}")
                return stored, reason, None
                
            return stored, reason, memory
            
        # Skip gating logic - embed and store directly
        embedding_response = await self.embedding_service.embed_texts([content])
        
        # Classify if no category provided
        if category is None:
            scores = await self.classifier_service.classify(content)
            best_category = max(scores.items(), key=lambda x: x[1])[0]
            category = best_category
        
        memory = Memory(
            content=content,
            category=category,
            source=source,
            embedding=embedding_response.embeddings[0],
            metadata=metadata or {},
        )
        
        stored_memory = await self.memory_service.add(memory)
        return True, "Gating skipped", stored_memory
        
    @track_latency()
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID.
        
        Args:
            memory_id: ID of the memory to get
            
        Returns:
            Optional[Memory]: Memory if found, None otherwise
        """
        return await self.memory_service.get(memory_id)
        
    @track_latency()
    async def search_memory(self, query: SearchQuery) -> SearchResult:
        """Search for memories.
        
        Args:
            query: Search query
            
        Returns:
            SearchResult: Search results
        """
        search_result = await self.memory_service.search(query)
        
        # Rerank if requested and available
        if query.rerank and search_result.memories and self.reranking_service:
            try:
                reranked_memories = await self.reranking_service.rerank(
                    query.query,
                    search_result.memories,
                    query.top_k
                )
                search_result.memories = reranked_memories
                search_result.reranked = True
            except Exception as e:
                logger.error(f"Error during reranking: {e}")
                
        return search_result
        
    @track_latency()
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            bool: True if the memory was deleted, False otherwise
        """
        return await self.memory_service.delete(memory_id)
        
    @track_latency()
    async def add_feedback(
        self, 
        text: str, 
        category: MemoryCategory,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add feedback for the classifier.
        
        Args:
            text: Text that was classified
            category: Correct category
            metadata: Optional metadata
        """
        if not self.feedback_service:
            logger.warning("Feedback service not available")
            return
            
        await self.feedback_service.log_feedback(text, category, metadata)
        
    @track_latency()
    async def retrain_classifier(self) -> None:
        """Retrain the classifier using collected feedback."""
        if not self.feedback_service:
            logger.warning("Feedback service not available")
            return
            
        await self.feedback_service.retrain_classifier() 