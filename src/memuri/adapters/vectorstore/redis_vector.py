"""
Redis Vector adapter for Redis-based vector operations.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from memuri.domain.interfaces import VectorStoreAdapter
from memuri.domain.models import Document, DocumentChunk


class RedisVectorAdapter(VectorStoreAdapter):
    """Adapter for Redis vector operations."""

    def __init__(
        self,
        redis_url: str,
        index_name: str = "memory_vectors",
        prefix: str = "doc:",
        embedding_dim: int = 1536,
        distance_metric: str = "COSINE",
    ):
        """Initialize Redis vector adapter.
        
        Args:
            redis_url: Redis connection URL
            index_name: Redis search index name
            prefix: Key prefix for documents
            embedding_dim: Dimension of embedding vectors
            distance_metric: Distance metric for vector search (COSINE, IP, L2)
        """
        self.redis_url = redis_url
        self.index_name = index_name
        self.prefix = prefix
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self.client = None

    async def initialize(self) -> None:
        """Initialize Redis connection and create index if it doesn't exist."""
        self.client = redis.from_url(self.redis_url, decode_responses=True)
        
        # Check if index exists
        try:
            await self.client.ft(self.index_name).info()
            index_exists = True
        except:
            index_exists = False
            
        # Create index if it doesn't exist
        if not index_exists:
            schema = [
                TextField("content"),
                TextField("document_id"),
                TextField("chunk_id"),
                TextField("metadata", as_name="metadata"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embedding_dim,
                        "DISTANCE_METRIC": self.distance_metric,
                    },
                ),
            ]
            
            await self.client.ft(self.index_name).create_index(
                schema,
                definition=IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH),
            )

    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()

    async def add(
        self, documents: List[Union[Document, DocumentChunk]], batch_size: int = 100
    ) -> List[str]:
        """Add documents to Redis vector store.
        
        Args:
            documents: List of documents or document chunks to add
            batch_size: Number of documents to add in each batch
            
        Returns:
            List of document IDs that were added
        """
        if not self.client:
            await self.initialize()
            
        doc_ids = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            pipe = self.client.pipeline()
            
            for doc in batch:
                # Handle both Document and DocumentChunk
                if isinstance(doc, Document):
                    doc_id = doc.id
                    chunk_id = "0"  # Use 0 for the full document
                    content = doc.content
                    embedding = doc.embedding
                    metadata = doc.metadata
                else:  # DocumentChunk
                    doc_id = doc.document_id
                    chunk_id = doc.id
                    content = doc.content
                    embedding = doc.embedding
                    metadata = doc.metadata
                
                # Convert numpy array to list if needed
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                # Key for Redis hash
                key = f"{self.prefix}{doc_id}:{chunk_id}"
                
                # Add to Redis
                pipe.hset(
                    key,
                    mapping={
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "content": content,
                        "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                        "metadata": json.dumps(metadata),
                    },
                )
                doc_ids.append(doc_id)
            
            # Execute pipeline
            await pipe.execute()
                
        return doc_ids

    async def search(
        self, 
        query_embedding: Union[List[float], np.ndarray],
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Union[Document, DocumentChunk], float]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Embedding vector to search for
            k: Number of results to return
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.client:
            await self.initialize()
            
        # Convert numpy array to bytes
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.astype(np.float32).tobytes()
        else:
            query_embedding = np.array(query_embedding, dtype=np.float32).tobytes()
        
        # Build filter query
        filter_query = "*"
        if filter_dict:
            filter_parts = []
            for key, value in filter_dict.items():
                if key == "document_id":
                    filter_parts.append(f"@document_id:{value}")
                else:
                    # For nested metadata, we would need more complex handling
                    filter_parts.append(f"@metadata:{key}:{value}")
            
            if filter_parts:
                filter_query = " ".join(filter_parts)
        
        # Execute vector search
        query_result = await self.client.ft(self.index_name).search(
            f"{filter_query}=>[KNN {k} @embedding $vec AS score]",
            {
                "vec": query_embedding,
            },
        )
        
        results = []
        for doc in query_result.docs:
            # Parse the results
            metadata = json.loads(doc.metadata) if doc.metadata else {}
            embedding_bytes = doc.embedding if hasattr(doc, "embedding") else None
            
            # Convert embedding bytes back to numpy array
            embedding = None
            if embedding_bytes:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # Create DocumentChunk
            chunk = DocumentChunk(
                document_id=doc.document_id,
                id=doc.chunk_id,
                content=doc.content,
                embedding=embedding,
                metadata=metadata,
            )
            
            # Score is 1 - distance for COSINE (convert to similarity)
            similarity = 1 - float(doc.score) if hasattr(doc, "score") else 1.0
            results.append((chunk, similarity))
            
        return results

    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents from Redis vector store.
        
        Args:
            document_ids: List of document IDs to delete
        """
        if not self.client:
            await self.initialize()
            
        # Get all keys matching the document IDs
        for doc_id in document_ids:
            # Get all keys for this document ID
            pattern = f"{self.prefix}{doc_id}:*"
            cursor = None
            while cursor != 0:
                cursor, keys = await self.client.scan(cursor or 0, match=pattern)
                if keys:
                    await self.client.delete(*keys) 