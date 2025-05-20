"""Memory orchestration services."""

import time
from typing import Dict, List, Optional, Any

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
        settings: Optional[MemuriSettings] = None,
    ):
        """Initialize the memory orchestrator.
        
        Args:
            memory_service: Memory storage service
            embedding_service: Embedding service
            reranking_service: Reranking service
            classifier_service: Classifier service
            feedback_service: Feedback service
            settings: Optional settings
        """
        self.memory_service = memory_service
        self.embedding_service = embedding_service
        self.reranking_service = reranking_service
        self.classifier_service = classifier_service
        self.feedback_service = feedback_service
        self.settings = settings
        
        logger.info("Initialized memory orchestrator")
    
    @track_latency()
    async def add_memory(
        self,
        content: str,
        category: Optional[MemoryCategory] = None,
        source: MemorySource = MemorySource.USER,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Add a memory to the store.
        
        Args:
            content: Content of the memory
            category: Optional category of the memory
            source: Source of the memory
            metadata: Optional metadata to associate with the memory
            
        Returns:
            Memory: The added memory
        """
        # Auto-classify if no category provided
        if category is None:
            classifier_results = await self.classifier_service.classify(content)
            # Get category with highest confidence
            category = max(classifier_results.items(), key=lambda x: x[1])[0]
        
        # Get embedding for content
        embedding_response = await self.embedding_service.embed_texts([content])
        embedding = embedding_response.embeddings[0]
        
        # Create memory object
        memory = Memory(
            content=content,
            category=category,
            source=source,
            embedding=embedding,
            metadata=metadata or {},
        )
        
        # Add to storage
        added_memory = await self.memory_service.add(memory)
        
        return added_memory
    
    @track_latency()
    async def search_memory(self, query: SearchQuery) -> SearchResult:
        """Search for memories.
        
        Args:
            query: Search query
            
        Returns:
            SearchResult: Search results
        """
        # Start timing
        start_time = time.time()
        
        # Search in memory service
        search_result = await self.memory_service.search(query)
        
        # Apply reranking if requested
        if query.rerank and search_result.memories:
            reranked_memories = await self.reranking_service.rerank(
                query=query.query,
                candidates=search_result.memories,
                top_k=query.top_k,
            )
            
            search_result.memories = reranked_memories
            search_result.reranked = True
        
        # Calculate search time
        search_result.search_time = time.time() - start_time
        
        return search_result 