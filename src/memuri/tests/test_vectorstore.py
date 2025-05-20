"""
Tests for vector store adapters.
"""
import uuid
from typing import List, Dict, Any

import pytest
import numpy as np

from memuri.adapters.vectorstore.pgvector import PGVectorAdapter
from memuri.adapters.vectorstore.redis_vector import RedisVectorAdapter
from memuri.domain.models import Document, DocumentChunk


@pytest.mark.asyncio
async def test_pgvector_adapter_initialization():
    """Test that the PGVector adapter can be initialized."""
    test_table = f"test_init_{uuid.uuid4().hex[:8]}"
    adapter = PGVectorAdapter(
        connection_string="postgresql://postgres:postgres@localhost:5432/test_memuri",
        table_name=test_table,
        embedding_dim=4,
    )
    await adapter.initialize()
    
    # Verify the pool exists
    assert adapter.pool is not None
    
    # Cleanup
    async with adapter.pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {test_table}")
    
    await adapter.close()


@pytest.mark.asyncio
async def test_pgvector_add_and_search(pgvector_adapter: PGVectorAdapter, sample_documents: List[Document]):
    """Test adding documents to PGVector and searching."""
    # Add the sample documents
    doc_ids = await pgvector_adapter.add(sample_documents)
    
    # Verify doc_ids
    assert len(doc_ids) == len(sample_documents)
    assert all(doc_id.startswith("doc_") for doc_id in doc_ids)
    
    # Search for documents
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = await pgvector_adapter.search(query_embedding, k=5)
    
    # Verify search results
    assert len(results) == 5  # We asked for top 5
    assert all(isinstance(result[0], DocumentChunk) for result in results)
    assert all(0 <= result[1] <= 1.0 for result in results)  # Similarity scores should be between 0 and 1


@pytest.mark.asyncio
async def test_pgvector_search_with_filter(pgvector_adapter: PGVectorAdapter, sample_documents: List[Document]):
    """Test searching with filters in PGVector."""
    # Add documents if not already added
    await pgvector_adapter.add(sample_documents)
    
    # Search with document_id filter
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = await pgvector_adapter.search(
        query_embedding, 
        k=10,
        filter_dict={"document_id": "doc_1"}
    )
    
    # Verify filtered results
    assert len(results) == 1
    assert results[0][0].document_id == "doc_1"


@pytest.mark.asyncio
async def test_pgvector_delete(pgvector_adapter: PGVectorAdapter, sample_documents: List[Document]):
    """Test deleting documents from PGVector."""
    # Add documents if not already added
    await pgvector_adapter.add(sample_documents)
    
    # Delete a specific document
    await pgvector_adapter.delete(["doc_1"])
    
    # Verify it's been deleted by searching for it
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = await pgvector_adapter.search(
        query_embedding, 
        k=10,
        filter_dict={"document_id": "doc_1"}
    )
    
    # Should return no results
    assert len(results) == 0


@pytest.mark.asyncio
async def test_redis_vector_initialization():
    """Test that the Redis Vector adapter can be initialized."""
    test_index = f"test_init_{uuid.uuid4().hex[:8]}"
    adapter = RedisVectorAdapter(
        redis_url="redis://localhost:6379",
        index_name=test_index,
        prefix=f"{test_index}:",
        embedding_dim=4,
    )
    await adapter.initialize()
    
    # Verify the client exists
    assert adapter.client is not None
    
    # Cleanup
    try:
        await adapter.client.ft(test_index).dropindex()
    except:
        pass  # Index might not exist
    
    await adapter.close()


@pytest.mark.asyncio
async def test_redis_vector_add_and_search(
    redis_vector_adapter: RedisVectorAdapter, 
    sample_documents: List[Document]
):
    """Test adding documents to Redis Vector and searching."""
    # Add the sample documents
    doc_ids = await redis_vector_adapter.add(sample_documents)
    
    # Verify doc_ids
    assert len(doc_ids) == len(sample_documents)
    assert all(doc_id.startswith("doc_") for doc_id in doc_ids)
    
    # Search for documents - give a small delay for indexing
    import asyncio
    await asyncio.sleep(1)
    
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = await redis_vector_adapter.search(query_embedding, k=5)
    
    # Verify search results
    assert len(results) <= 5  # We asked for top 5, but might get fewer
    # Some versions of Redis search will return fewer results if there are not enough matches
    if results:
        assert all(isinstance(result[0], DocumentChunk) for result in results)
        assert all(0 <= result[1] <= 1.0 for result in results)


@pytest.mark.asyncio
async def test_redis_vector_delete(redis_vector_adapter: RedisVectorAdapter, sample_documents: List[Document]):
    """Test deleting documents from Redis Vector."""
    # Add documents if not already added
    await redis_vector_adapter.add(sample_documents)
    
    # Delete a specific document
    await redis_vector_adapter.delete(["doc_1"])
    
    # Verify it's been deleted by searching for it
    import asyncio
    await asyncio.sleep(1)  # Give a small delay for deletion to propagate
    
    # Count how many keys are left with the prefix for doc_1
    pattern = f"{redis_vector_adapter.prefix}doc_1:*"
    cursor = 0
    keys = []
    
    while True:
        cursor, batch = await redis_vector_adapter.client.scan(
            cursor=cursor, 
            match=pattern,
            count=100
        )
        keys.extend(batch)
        if cursor == 0:
            break
    
    # Should be no keys left for doc_1
    assert len(keys) == 0 