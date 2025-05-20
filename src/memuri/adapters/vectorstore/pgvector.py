"""
PGVector adapter for PostgreSQL-based vector operations.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import json
import datetime

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from memuri.domain.interfaces import VectorStoreAdapter, MemoryService
from memuri.domain.models import Document, DocumentChunk, Memory, MemoryCategory, MemorySource, SearchQuery, SearchResult, ScoredMemory
from memuri.core.config import VectorStoreSettings


class PGVectorAdapter(VectorStoreAdapter):
    """Adapter for PostgreSQL pgvector extension."""

    def __init__(
        self,
        connection_string: str,
        table_name: str = "memory_vectors",
        embedding_dim: int = 1536,
        max_conn: int = 10,
    ):
        """Initialize pgvector adapter.
        
        Args:
            connection_string: PostgreSQL connection string
            table_name: Table name for storing vectors
            embedding_dim: Dimension of embedding vectors
            max_conn: Maximum number of connections in the pool
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.max_conn = max_conn
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        """Initialize connection pool and create tables if they don't exist."""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=2,
            max_size=self.max_conn,
            setup=self._setup_connection,
        )
        
        # Create table if it doesn't exist
        async with self.pool.acquire() as conn:
            # First, create the vector extension if it doesn't exist
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception as e:
                logger.warning(f"Failed to create vector extension: {e}. Extension might already exist or requires admin privileges.")
            
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self.embedding_dim}) NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(document_id, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS {self.table_name}_document_id_idx 
                    ON {self.table_name}(document_id);
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                    ON {self.table_name} USING ivfflat (embedding vector_cosine_ops);
                """
            )

    async def _setup_connection(self, conn: asyncpg.Connection) -> None:
        """Set up connection with vector extension."""
        await register_vector(conn)

    async def close(self) -> None:
        """Close database connections."""
        if self.pool:
            await self.pool.close()

    async def add(
        self, documents: List[Union[Document, DocumentChunk]], batch_size: int = 100
    ) -> List[str]:
        """Add documents to vector store.
        
        Args:
            documents: List of documents or document chunks to add
            batch_size: Number of documents to add in each batch
            
        Returns:
            List of document IDs that were added
        """
        if not self.pool:
            await self.initialize()
            
        doc_ids = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            async with self.pool.acquire() as conn:
                async with conn.transaction():
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
                            
                        # Add to database
                        await conn.execute(
                            f"""
                            INSERT INTO {self.table_name} 
                                (document_id, chunk_id, content, embedding, metadata)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (document_id, chunk_id) 
                            DO UPDATE SET 
                                content = EXCLUDED.content, 
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata
                            """,
                            doc_id,
                            chunk_id,
                            content,
                            embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                            metadata,
                        )
                        doc_ids.append(doc_id)
                        
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
        if not self.pool:
            await self.initialize()
            
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Build WHERE clause for filtering
        filter_conditions = []
        filter_params = [query_embedding, k]
        param_idx = 3
        
        if filter_dict:
            for key, value in filter_dict.items():
                if key == 'document_id':
                    filter_conditions.append(f"document_id = ${param_idx}")
                    filter_params.append(value)
                    param_idx += 1
                else:
                    filter_conditions.append(f"metadata->>'${key}' = ${param_idx}")
                    filter_params.append(value)
                    param_idx += 1
        
        filter_clause = f"WHERE {' AND '.join(filter_conditions)}" if filter_conditions else ""
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT 
                    document_id, 
                    chunk_id, 
                    content, 
                    embedding, 
                    metadata, 
                    1 - (embedding <=> $1) as similarity
                FROM {self.table_name}
                {filter_clause}
                ORDER BY similarity DESC
                LIMIT $2
                """,
                *filter_params
            )
        
        results = []
        for row in rows:
            chunk = DocumentChunk(
                document_id=row['document_id'],
                id=row['chunk_id'],
                content=row['content'],
                embedding=np.array(row['embedding']),
                metadata=row['metadata'] or {},
            )
            results.append((chunk, row['similarity']))
            
        return results
            
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents from vector store.
        
        Args:
            document_ids: List of document IDs to delete
        """
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                DELETE FROM {self.table_name}
                WHERE document_id = ANY($1::text[])
                """,
                document_ids,
            ) 

# New class implementing MemoryService interface for the factory pattern
class PgVectorStore(MemoryService):
    """PostgreSQL with pgvector as a MemoryService."""
    
    def __init__(self, settings: VectorStoreSettings):
        """Initialize pgvector memory service.
        
        Args:
            settings: Vector store settings
        """
        self.connection_string = settings.connection_string
        self.collection_name = settings.collection_name or "memories"
        self.dimensions = settings.dimensions or 1536
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self) -> None:
        """Initialize connection pool and create tables if they don't exist."""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=2,
            max_size=10,
            setup=self._setup_connection,
        )
        
        # Create table if it doesn't exist
        async with self.pool.acquire() as conn:
            # First, create the vector extension if it doesn't exist
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception as e:
                logger.warning(f"Failed to create vector extension: {e}. Extension might already exist or requires admin privileges.")
            
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    source TEXT NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS {self.collection_name}_category_idx 
                    ON {self.collection_name}(category);
                CREATE INDEX IF NOT EXISTS {self.collection_name}_embedding_idx 
                    ON {self.collection_name} USING ivfflat (embedding vector_cosine_ops);
                """
            )
            
    async def _setup_connection(self, conn: asyncpg.Connection) -> None:
        """Set up connection with vector extension."""
        await register_vector(conn)
        
    async def add(self, memory: Memory) -> Memory:
        """Add a memory to the store.
        
        Args:
            memory: Memory to add
            
        Returns:
            Memory: Added memory with ID and any other fields populated
        """
        if not self.pool:
            await self.initialize()
            
        # Generate ID if not provided
        if not memory.id:
            memory.id = str(uuid.uuid4())
            
        # Convert embedding to list if needed
        embedding = memory.embedding
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            
        # Convert metadata to JSON if needed
        metadata = memory.metadata or {}
            
        # Add to database
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.collection_name} 
                    (id, content, category, source, embedding, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) 
                DO UPDATE SET 
                    content = EXCLUDED.content, 
                    category = EXCLUDED.category,
                    source = EXCLUDED.source,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                memory.id,
                memory.content,
                memory.category.value,
                memory.source.value,
                embedding,
                json.dumps(metadata),
                memory.created_at,
            )
            
        return memory
        
    async def add_batch(self, memories: List[Memory]) -> List[Memory]:
        """Add a batch of memories to the store.
        
        Args:
            memories: Memories to add
            
        Returns:
            List[Memory]: Added memories with IDs and any other fields populated
        """
        if not self.pool:
            await self.initialize()
            
        for memory in memories:
            if not memory.id:
                memory.id = str(uuid.uuid4())
                
        # Add each memory one by one
        for memory in memories:
            await self.add(memory)
            
        return memories
        
    async def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID.
        
        Args:
            memory_id: ID of the memory to get
            
        Returns:
            Optional[Memory]: Memory if found, None otherwise
        """
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, content, category, source, embedding, metadata, created_at
                FROM {self.collection_name}
                WHERE id = $1
                """,
                memory_id,
            )
            
        if not row:
            return None
            
        # Convert row to Memory
        return Memory(
            id=row['id'],
            content=row['content'],
            category=MemoryCategory(row['category']),
            source=row['source'],
            embedding=row['embedding'],
            metadata=json.loads(row['metadata']),
            created_at=row['created_at'],
        )
        
    async def update(self, memory: Memory) -> Memory:
        """Update a memory in the store.
        
        Args:
            memory: Memory to update
            
        Returns:
            Memory: Updated memory
        """
        if not self.pool:
            await self.initialize()
            
        # Add/update memory
        return await self.add(memory)
        
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            bool: True if the memory was deleted, False otherwise
        """
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.collection_name}
                WHERE id = $1
                """,
                memory_id,
            )
            
        # Check if a row was deleted
        return "DELETE 1" in result
        
    async def search(self, query: SearchQuery) -> SearchResult:
        """Search for memories.
        
        Args:
            query: Search query
            
        Returns:
            SearchResult: Search results
        """
        if not self.pool:
            await self.initialize()
            
        # Get query embedding
        embedding_service = await self._get_embedding_service()
        embedding_response = await embedding_service.embed_texts([query.query])
        query_embedding = embedding_response.embeddings[0]
        
        # Search by vector
        scored_memories = await self.search_by_vector(
            vector=query_embedding,
            category=query.category,
            top_k=query.top_k,
            metadata_filters=query.metadata_filters,
        )
        
        return SearchResult(
            memories=scored_memories,
            query=query.query,
            total_found=len(scored_memories),
            search_time=0.0,  # TODO: Track search time
            reranked=False,  # TODO: Implement reranking
        )
        
    async def search_by_vector(
        self, 
        vector: List[float], 
        category: Optional[MemoryCategory] = None,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredMemory]:
        """Search for memories by vector similarity.
        
        Args:
            vector: Query vector
            category: Optional category filter
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
            
        Returns:
            List[ScoredMemory]: Scored memories
        """
        if not self.pool:
            await self.initialize()
            
        # Build WHERE clause for filtering
        filter_conditions = []
        filter_params = [vector, top_k]
        param_idx = 3
        
        # Add category filter if provided
        if category:
            filter_conditions.append(f"category = ${param_idx}")
            filter_params.append(category.value)
            param_idx += 1
            
        # Add metadata filters if provided
        if metadata_filters:
            for key, value in metadata_filters.items():
                filter_conditions.append(f"metadata->>'${key}' = ${param_idx}")
                filter_params.append(value)
                param_idx += 1
                
        filter_clause = f"WHERE {' AND '.join(filter_conditions)}" if filter_conditions else ""
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT 
                    id, 
                    content, 
                    category,
                    source,
                    embedding, 
                    metadata,
                    created_at,
                    1 - (embedding <=> $1) as similarity
                FROM {self.collection_name}
                {filter_clause}
                ORDER BY similarity DESC
                LIMIT $2
                """,
                *filter_params
            )
            
        # Convert rows to ScoredMemory objects
        results = []
        for row in rows:
            memory = Memory(
                id=row['id'],
                content=row['content'],
                category=MemoryCategory(row['category']),
                source=row['source'],
                embedding=row['embedding'],
                metadata=json.loads(row['metadata']),
                created_at=row['created_at'],
            )
            scored_memory = ScoredMemory(
                memory=memory,
                score=row['similarity'],
            )
            results.append(scored_memory)
            
        return results
        
    async def search_by_text(
        self, 
        text: str,
        category: Optional[MemoryCategory] = None,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredMemory]:
        """Search for memories by text.
        
        This method embeds the query text and then searches by vector.
        
        Args:
            text: Query text
            category: Optional category filter
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
            
        Returns:
            List[ScoredMemory]: Scored memories
        """
        if not self.pool:
            await self.initialize()
            
        # Get embedding for text query
        embedding_service = await self._get_embedding_service()
        embedding_response = await embedding_service.embed_texts([text])
        query_embedding = embedding_response.embeddings[0]
        
        # Search by vector
        return await self.search_by_vector(
            vector=query_embedding,
            category=category,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        
    async def count(
        self, 
        category: Optional[MemoryCategory] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Count memories matching the given filters.
        
        Args:
            category: Optional category filter
            metadata_filters: Optional metadata filters
            
        Returns:
            int: Number of matching memories
        """
        if not self.pool:
            await self.initialize()
            
        # Build WHERE clause for filtering
        filter_conditions = []
        filter_params = []
        param_idx = 1
        
        # Add category filter if provided
        if category:
            filter_conditions.append(f"category = ${param_idx}")
            filter_params.append(category.value)
            param_idx += 1
            
        # Add metadata filters if provided
        if metadata_filters:
            for key, value in metadata_filters.items():
                filter_conditions.append(f"metadata->>'${key}' = ${param_idx}")
                filter_params.append(value)
                param_idx += 1
                
        filter_clause = f"WHERE {' AND '.join(filter_conditions)}" if filter_conditions else ""
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT COUNT(*) as count
                FROM {self.collection_name}
                {filter_clause}
                """,
                *filter_params
            )
            
        return row['count']
        
    async def _get_embedding_service(self):
        """Get embedding service for text queries.
        
        Returns:
            EmbeddingService: Embedding service
        """
        from memuri.factory import EmbedderFactory
        return EmbedderFactory.create() 