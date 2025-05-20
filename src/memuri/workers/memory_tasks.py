"""
Celery tasks for memory management operations.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

from memuri.domain.models import Document, DocumentChunk
from memuri.factory import VectorStoreFactory, CacheFactory, DbFactory
from memuri.core.config import settings
from memuri.services.rerank import CrossEncoderReranker

# Configure logging
logger = logging.getLogger(__name__)

# Get Celery app from celery_app.py or create a new one
try:
    from memuri.workers.celery_app import app
except ImportError:
    # Configure Celery
    redis_url = settings.redis_url
    app = Celery(
        "memuri", 
        broker=f"{redis_url}/0",
        backend=f"{redis_url}/1",
    )
    app.conf.task_serializer = "json"
    app.conf.result_serializer = "json"
    app.conf.accept_content = ["json"]
    app.conf.task_routes = {
        "memuri.workers.embed_tasks.*": {"queue": "embedding"},
        "memuri.workers.memory_tasks.*": {"queue": "memory"},
    }

# Global state for connection pools
vector_store = None
cache = None
db = None
reranker = None


@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize connections when worker starts."""
    global vector_store, cache, db, reranker
    
    # Create connections asynchronously
    loop = asyncio.get_event_loop()
    
    async def init_connections():
        # Initialize vector store
        vector_store_factory = VectorStoreFactory()
        vs = await vector_store_factory.create(settings.vector_store_provider)
        
        # Initialize cache
        cache_factory = CacheFactory()
        c = await cache_factory.create(settings.cache_provider)
        
        # Initialize database
        db_factory = DbFactory()
        d = await db_factory.create(settings.db_provider)
        
        # Initialize reranker
        r = CrossEncoderReranker(
            model_name=settings.reranker_model,
            device=settings.reranker_device,
            batch_size=settings.reranker_batch_size,
        )
        await r.initialize()
        
        return vs, c, d, r
    
    vector_store, cache, db, reranker = loop.run_until_complete(init_connections())
    logger.info("Memory worker initialized with vector store, cache, and database")


@worker_process_shutdown.connect
def shutdown_worker(**kwargs):
    """Close connections when worker shuts down."""
    global vector_store, cache, db, reranker
    
    # Close connections asynchronously
    loop = asyncio.get_event_loop()
    
    async def close_connections():
        if vector_store:
            await vector_store.close()
        if cache:
            await cache.close()
        if db:
            await db.close()
        if reranker:
            await reranker.close()
    
    loop.run_until_complete(close_connections())
    logger.info("Memory worker connections closed")


@app.task(name="memuri.workers.memory_tasks.rerank_results")
def rerank_results(
    query: str,
    results: List[Dict],
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
) -> List[Tuple[Dict, float]]:
    """Rerank search results using cross-encoder.
    
    Args:
        query: Search query
        results: List of document dictionaries from initial retrieval
        alpha: Weight for cross-encoder score
        beta: Weight for time decay factor
        gamma: Weight for metadata score
        
    Returns:
        Reranked list of (document, score) tuples
    """
    global reranker
    
    if not reranker:
        logger.error("Reranker not initialized")
        raise RuntimeError("Reranker not initialized")
    
    # Convert dictionaries to Document/DocumentChunk objects
    documents = []
    for item in results:
        if "document_id" in item:
            # It's a document chunk
            chunk = DocumentChunk(
                document_id=item["document_id"],
                id=item["id"],
                content=item["content"],
                metadata=item.get("metadata", {})
            )
            documents.append(chunk)
        else:
            # It's a full document
            doc = Document(
                id=item["id"],
                content=item["content"],
                metadata=item.get("metadata", {})
            )
            documents.append(doc)
    
    # Run the reranking asynchronously
    loop = asyncio.get_event_loop()
    
    async def process_reranking():
        # Get cross-encoder scores
        cross_scores = await reranker.score_batch(query, documents)
        
        # Calculate time decay for each document
        now = time.time()
        time_decay_scores = []
        for doc in documents:
            # Get created_at timestamp from metadata or use current time
            created_at = doc.metadata.get("created_at", now)
            if isinstance(created_at, str):
                try:
                    # Try to parse ISO format
                    from datetime import datetime
                    created_at = datetime.fromisoformat(created_at).timestamp()
                except (ValueError, TypeError):
                    created_at = now
            
            # Calculate time decay factor (newer documents score higher)
            # Normalize to [0,1] with exponential decay
            age_days = (now - created_at) / (24 * 3600)  # age in days
            decay_factor = max(0.0, min(1.0, 2.71828 ** (-0.05 * age_days)))  # exp(-0.05 * age_days)
            time_decay_scores.append(decay_factor)
        
        # Calculate metadata scores (e.g., based on category relevance)
        metadata_scores = []
        for doc in documents:
            # Example: score based on category match or importance flag
            category_match = 1.0 if doc.metadata.get("category") == "IMPORTANT" else 0.5
            is_favorite = 1.0 if doc.metadata.get("favorite", False) else 0.0
            metadata_score = 0.7 * category_match + 0.3 * is_favorite
            metadata_scores.append(metadata_score)
        
        # Combine scores with weights
        final_scores = []
        for i, doc in enumerate(documents):
            # Apply weighted score combination
            score = (
                alpha * cross_scores[i] +
                beta * time_decay_scores[i] +
                gamma * metadata_scores[i]
            )
            final_scores.append(score)
        
        # Create result tuples and sort by final score
        reranked_results = [
            (doc.dict() if hasattr(doc, "dict") else vars(doc), score)
            for doc, score in zip(documents, final_scores)
        ]
        return sorted(reranked_results, key=lambda x: x[1], reverse=True)
    
    return loop.run_until_complete(process_reranking())


@app.task(name="memuri.workers.memory_tasks.cleanup_memories")
def cleanup_memories(
    older_than_days: int = 90,
    max_documents: int = 10000,
    categories_to_keep: Optional[List[str]] = None,
) -> int:
    """Clean up old memories to prevent vector store bloat.
    
    Args:
        older_than_days: Remove memories older than this many days
        max_documents: Maximum number of documents to keep
        categories_to_keep: Categories to always keep
        
    Returns:
        Number of documents deleted
    """
    global vector_store, db
    
    if not vector_store or not db:
        logger.error("Vector store or database not initialized")
        raise RuntimeError("Vector store or database not initialized")
    
    # Run the cleanup asynchronously
    loop = asyncio.get_event_loop()
    
    async def process_cleanup():
        # Get document count from database
        count_query = "SELECT COUNT(*) FROM memory_entries"
        total_count = await db.fetch_val(count_query)
        
        # If under max limit and no time-based cleanup needed, exit early
        if total_count <= max_documents and older_than_days <= 0:
            return 0
            
        # Build query to find documents to delete
        conditions = []
        params = []
        
        # Add time-based condition
        if older_than_days > 0:
            conditions.append("created_at < NOW() - INTERVAL $1 DAY")
            params.append(older_than_days)
            
        # Add category exclusion if provided
        if categories_to_keep:
            placeholders = ", ".join(f"${i+1+len(params)}" for i in range(len(categories_to_keep)))
            conditions.append(f"category NOT IN ({placeholders})")
            params.extend(categories_to_keep)
            
        # Build final query
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        query = f"""
        SELECT memory_id 
        FROM memory_entries 
        WHERE {where_clause}
        ORDER BY created_at ASC
        """
        
        # If over max count, limit deletion to the difference
        if total_count > max_documents:
            delete_count = total_count - max_documents
            query += f" LIMIT ${len(params) + 1}"
            params.append(delete_count)
            
        # Get IDs to delete
        rows = await db.fetch(query, *params)
        doc_ids = [row["memory_id"] for row in rows]
        
        if not doc_ids:
            return 0
        
        # Delete from vector store
        await vector_store.delete(doc_ids)
        
        # Delete from database
        delete_query = "DELETE FROM memory_entries WHERE memory_id = ANY($1::text[])"
        await db.execute(delete_query, doc_ids)
        
        logger.info(f"Cleaned up {len(doc_ids)} old memories")
        return len(doc_ids)
    
    return loop.run_until_complete(process_cleanup())


@app.task(name="memuri.workers.memory_tasks.update_feedback_classifier")
def update_feedback_classifier() -> Dict[str, Any]:
    """Retrain classifier with new feedback data.
    
    Returns:
        Dictionary with training metrics
    """
    global db
    
    if not db:
        logger.error("Database not initialized")
        raise RuntimeError("Database not initialized")
    
    # Run the training asynchronously
    loop = asyncio.get_event_loop()
    
    async def process_training():
        # Get feedback data that hasn't been used for training yet
        query = """
        SELECT content, category 
        FROM user_feedback 
        WHERE used_for_training = FALSE
        """
        rows = await db.fetch(query)
        
        if not rows:
            return {"status": "no_new_data", "samples": 0}
        
        # TODO: Implement actual classifier retraining
        # This is a placeholder for the real implementation
        # In a real implementation, we would:
        # 1. Load the current classifier model
        # 2. Fine-tune it with the new data
        # 3. Save the updated model
        # 4. Update training metrics
        
        # Get counts by category
        categories = {}
        for row in rows:
            cat = row["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        # Mark feedback as used for training
        update_query = """
        UPDATE user_feedback 
        SET used_for_training = TRUE 
        WHERE used_for_training = FALSE
        """
        await db.execute(update_query)
        
        return {
            "status": "success",
            "samples": len(rows),
            "categories": categories,
            "timestamp": time.time()
        }
    
    return loop.run_until_complete(process_training()) 