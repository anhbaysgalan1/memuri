"""
PGVector adapter for PostgreSQL-based vector operations.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from memuri.domain.interfaces import VectorStoreAdapter
from memuri.domain.models import Document, DocumentChunk


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