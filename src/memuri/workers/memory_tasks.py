"""
Memory-related tasks for Celery workers.

This module provides Celery tasks for asynchronous memory operations.
"""

from typing import Dict, Any, Optional
import logging
import datetime

from celery import shared_task
from memuri.domain.models import Memory, MemoryCategory, MemorySource, SearchQuery
from memuri.factory import VectorStoreFactory, ClassifierFactory
from memuri.core.config import get_settings
from memuri.core.text_utils import clean_text, normalize_text_for_embedding

logger = logging.getLogger(__name__)


@shared_task(
    name="memuri.workers.memory_tasks.add_memory",
    bind=True,
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def add_memory(
    self,
    content: str,
    category: str = "GENERAL",
    source: str = "USER",
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Add a memory to the vector store asynchronously.

    Args:
        content: Text content of the memory
        category: Memory category (defaults to GENERAL)
        source: Source of the memory (defaults to USER)
        user_id: Optional user ID for the memory
        metadata: Optional metadata for the memory

    Returns:
        Dict with memory ID and status
    """
    # Clean and normalize text
    cleaned_content = normalize_text_for_embedding(content)

    # Get settings and create vector store
    settings = get_settings()
    vector_store = VectorStoreFactory.create(
        provider=settings.vector_store.provider,
        settings=settings.vector_store
    )

    # Create memory object
    try:
        memory = Memory(
            content=cleaned_content,
            category=MemoryCategory(category),
            source=MemorySource(source),
            user_id=user_id,
            metadata=metadata or {},
            created_at=datetime.datetime.now().isoformat(),
        )

        # Add to vector store
        memory_id = vector_store.add_memory(memory)
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "category": category,
        }
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise self.retry(exc=e)


@shared_task(
    name="memuri.workers.memory_tasks.search_memories",
    bind=True,
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def search_memories(
    self,
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    min_score: float = 0.7,
) -> Dict[str, Any]:
    """
    Search for memories asynchronously.

    Args:
        query: Search query text
        top_k: Number of results to return
        category: Optional category filter
        user_id: Optional user ID filter
        metadata_filter: Optional metadata filter
        min_score: Minimum similarity score threshold

    Returns:
        Dict with search results
    """
    # Clean and normalize query
    cleaned_query = normalize_text_for_embedding(query)

    # Get settings and create vector store
    settings = get_settings()
    vector_store = VectorStoreFactory.create(
        provider=settings.vector_store.provider,
        settings=settings.vector_store
    )

    # Create search query
    search_query = SearchQuery(
        query=cleaned_query,
        top_k=top_k,
        category=MemoryCategory(category) if category else None,
        user_id=user_id,
        metadata_filter=metadata_filter,
        min_score=min_score,
    )

    try:
        # Execute search
        results = vector_store.search(search_query)
        
        # Format results
        formatted_results = []
        for result in results.memories:
            formatted_results.append({
                "memory_id": result.memory.id,
                "content": result.memory.content,
                "category": result.memory.category.value,
                "score": result.score,
                "created_at": result.memory.created_at,
                "metadata": result.memory.metadata,
            })
        
        return {
            "status": "success",
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
        }
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise self.retry(exc=e)


@shared_task(
    name="memuri.workers.memory_tasks.classify_text",
    bind=True,
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def classify_text(
    self,
    text: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Classify text into memory categories.

    Args:
        text: Text to classify
        threshold: Confidence threshold for classification

    Returns:
        Dict with classification results
    """
    # Clean text
    cleaned_text = clean_text(text)

    # Get settings and create classifier
    classifier = ClassifierFactory.create(
        provider="keyword",  # Default to keyword classifier
    )

    try:
        # Classify text
        classification = classifier.classify(cleaned_text)
        
        # Filter results by threshold
        filtered_results = {
            category: score
            for category, score in classification.items()
            if score >= threshold
        }
        
        # Get best category
        best_category = max(
            filtered_results.items(),
            key=lambda x: x[1],
            default=(None, 0)
        )
        
        return {
            "status": "success",
            "text": text,
            "classifications": filtered_results,
            "best_category": best_category[0],
            "confidence": best_category[1],
        }
    except Exception as e:
        logger.error(f"Error classifying text: {e}")
        raise self.retry(exc=e)


@shared_task(
    name="memuri.workers.memory_tasks.delete_memory",
    bind=True,
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def delete_memory(
    self,
    memory_id: str,
) -> Dict[str, Any]:
    """
    Delete a memory from the vector store.

    Args:
        memory_id: ID of the memory to delete

    Returns:
        Dict with status information
    """
    # Get settings and create vector store
    settings = get_settings()
    vector_store = VectorStoreFactory.create(
        provider=settings.vector_store.provider,
        settings=settings.vector_store
    )
    try:
        # Delete memory
        success = vector_store.delete(memory_id)        
        return {
            "status": "success" if success else "error",
            "memory_id": memory_id,
            "deleted": success,
        }
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise self.retry(exc=e) 