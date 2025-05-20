"""
Celery tasks for embedding operations.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Union

import numpy as np
from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

from memuri.domain.models import Document, DocumentChunk
from memuri.factory import EmbedderFactory, VectorStoreFactory
from memuri.core.config import settings

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
embedder = None
vector_store = None


@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize connections when worker starts."""
    global embedder, vector_store
    
    # Create embedder and vector store asynchronously
    loop = asyncio.get_event_loop()
    
    async def init_connections():
        # Initialize embedder
        embedder_factory = EmbedderFactory()
        embedder = await embedder_factory.create(settings.embedder_provider)
        
        # Initialize vector store
        vector_store_factory = VectorStoreFactory()
        vector_store = await vector_store_factory.create(settings.vector_store_provider)
        
        return embedder, vector_store
    
    embedder, vector_store = loop.run_until_complete(init_connections())
    logger.info("Worker initialized with embedder and vector store")


@worker_process_shutdown.connect
def shutdown_worker(**kwargs):
    """Close connections when worker shuts down."""
    global embedder, vector_store
    
    # Close connections asynchronously
    loop = asyncio.get_event_loop()
    
    async def close_connections():
        if embedder:
            await embedder.close()
        if vector_store:
            await vector_store.close()
    
    loop.run_until_complete(close_connections())
    logger.info("Worker connections closed")


@app.task(name="memuri.workers.embed_tasks.embed_batch", rate_limit="100/m")
def embed_batch(items: List[Dict], batch_size: int = 32) -> List[str]:
    """Embed a batch of items and store in vector store.
    
    Args:
        items: List of document dictionaries to embed
        batch_size: Number of documents to embed at once
        
    Returns:
        List of document IDs that were processed
    """
    global embedder, vector_store
    
    if not embedder or not vector_store:
        logger.error("Embedder or vector store not initialized")
        raise RuntimeError("Embedder or vector store not initialized")
    
    # Convert dictionaries to Document/DocumentChunk objects
    documents = []
    for item in items:
        if "document_id" in item:
            # It's a document chunk
            documents.append(DocumentChunk(
                document_id=item["document_id"],
                id=item["id"],
                content=item["content"],
                metadata=item.get("metadata", {})
            ))
        else:
            # It's a full document
            documents.append(Document(
                id=item["id"],
                content=item["content"],
                metadata=item.get("metadata", {})
            ))
    
    # Run the embedding and storage asynchronously
    loop = asyncio.get_event_loop()
    
    async def process_batch():
        # Embed the documents
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Get the content for embedding
            texts = [doc.content for doc in batch]
            
            # Generate embeddings
            embeddings = await embedder.embed_batch(texts)
            
            # Update documents with embeddings
            for j, doc in enumerate(batch):
                doc.embedding = embeddings[j]
            
            # Store in vector database
            await vector_store.add(batch)
            
            logger.info(f"Embedded and stored batch of {len(batch)} documents")
        
        # Return the document IDs
        return [doc.id if isinstance(doc, Document) else doc.document_id for doc in documents]
    
    return loop.run_until_complete(process_batch())


@app.task(name="memuri.workers.embed_tasks.embed_document", rate_limit="100/m")
def embed_document(
    content: str, 
    document_id: str, 
    chunk_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> str:
    """Embed a single document and store in vector store.
    
    Args:
        content: Text content to embed
        document_id: Document ID
        chunk_id: Optional chunk ID (if it's a document chunk)
        metadata: Optional metadata
        
    Returns:
        Document ID
    """
    global embedder, vector_store
    
    if not embedder or not vector_store:
        logger.error("Embedder or vector store not initialized")
        raise RuntimeError("Embedder or vector store not initialized")
    
    # Create document object
    if chunk_id:
        doc = DocumentChunk(
            document_id=document_id,
            id=chunk_id,
            content=content,
            metadata=metadata or {}
        )
    else:
        doc = Document(
            id=document_id,
            content=content,
            metadata=metadata or {}
        )
    
    # Run the embedding and storage asynchronously
    loop = asyncio.get_event_loop()
    
    async def process_document():
        # Generate embedding
        embedding = await embedder.embed(content)
        doc.embedding = embedding
        
        # Store in vector database
        await vector_store.add([doc])
        
        return document_id
    
    return loop.run_until_complete(process_document()) 