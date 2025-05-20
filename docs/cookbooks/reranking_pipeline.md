# Reranking Pipeline Cookbook

This cookbook demonstrates how to leverage Memuri's advanced reranking capabilities to improve the relevance of memory search results.

## Understanding Reranking

Memuri uses a multi-stage retrieval process:

1. **Initial Retrieval**: Fast vector similarity search to find candidate memories (top-N)
2. **Reranking**: More sophisticated algorithms to reorder candidates for better relevance
3. **Score Fusion**: Combining multiple signals (semantic similarity, recency, metadata)

This approach balances speed and accuracy by using a fast first-pass retrieval, then applying more computationally intensive reranking only to the top candidates.

## Basic Reranking

By default, Memuri applies reranking automatically with balanced settings:

```python
from memuri import Memuri

memory = Memuri()
await memory.initialize()

# Standard search (applies default reranking)
results = await memory.search_memory(
    query="What should I remember about tomorrow's meeting?",
    limit=5
)
```

## Customizing Reranker Settings

You can customize the reranking behavior:

```python
from memuri import Memuri
from memuri.core.config import RerankerConfig

# Configure custom reranking weights
reranker_config = RerankerConfig(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Lighter, faster model
    alpha=0.8,  # Cross-encoder semantic score weight (0.0-1.0)
    beta=0.1,   # Time decay factor weight (0.0-1.0)
    gamma=0.1,  # Metadata relevance weight (0.0-1.0)
    initial_k=50,  # Retrieve top-50 for reranking
    batch_size=16  # Process in batches of 16
)

memory = Memuri(reranker=reranker_config)
await memory.initialize()

# Search with custom reranking
results = await memory.search_memory("What's on my schedule?")
```

## Disabling Reranking

For cases where speed is more important than precision:

```python
# Search without reranking
results = await memory.search_memory(
    query="Quick search term",
    rerank=False
)
```

## Advanced Reranking Control

### Custom Metadata Score Function

You can provide a custom scoring function for metadata:

```python
from memuri import Memuri
from memuri.core.config import RerankerConfig
from typing import Dict, Any

# Custom metadata scoring function
async def my_metadata_scorer(query: str, metadata: Dict[str, Any]) -> float:
    """Custom scoring for metadata fields"""
    score = 0.0
    
    # Boost items with matching tags
    if "tags" in metadata:
        query_terms = set(query.lower().split())
        metadata_tags = set(tag.lower() for tag in metadata["tags"])
        tag_overlap = query_terms.intersection(metadata_tags)
        if tag_overlap:
            score += 0.2 * len(tag_overlap)
    
    # Boost by importance level
    if "importance" in metadata:
        score += 0.1 * min(metadata["importance"], 5)  # Cap at 0.5
        
    return min(score, 1.0)  # Normalize to 0-1

# Configure with custom metadata scorer
reranker_config = RerankerConfig(
    alpha=0.6,
    beta=0.1,
    gamma=0.3,  # Higher weight for metadata scoring
    metadata_scorer=my_metadata_scorer
)

memory = Memuri(reranker=reranker_config)
```

### Custom Time Decay Function

Configure how the recency of memories affects ranking:

```python
from memuri import Memuri
from memuri.core.config import RerankerConfig
import math
import time

# Custom time decay function
async def custom_time_decay(timestamp: float, now: float = None) -> float:
    """Custom time decay with step function for recent items"""
    if now is None:
        now = time.time()
        
    age_hours = (now - timestamp) / 3600
    
    # Step function
    if age_hours < 24:
        return 1.0  # Full weight for last 24 hours
    elif age_hours < 72:
        return 0.8  # High weight for 1-3 days
    elif age_hours < 168:
        return 0.6  # Medium weight for 3-7 days
    else:
        # Logarithmic decay for older items
        return max(0.1, 0.5 - (0.2 * math.log10(age_hours / 168)))

# Configure with custom time decay
reranker_config = RerankerConfig(
    alpha=0.7,
    beta=0.2,  # Time decay weight
    gamma=0.1,
    time_decay_fn=custom_time_decay
)

memory = Memuri(reranker=reranker_config)
```

## Exploring Reranking Models

Memuri supports different cross-encoder models for reranking:

```python
from memuri import Memuri
from memuri.core.config import RerankerConfig

# Option 1: Fast, lightweight model
fast_reranker = RerankerConfig(
    model="cross-encoder/ms-marco-MiniLM-L-4-v2",
    device="cpu"
)

# Option 2: Balanced model (default)
balanced_reranker = RerankerConfig(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"
)

# Option 3: High-quality model (slower)
quality_reranker = RerankerConfig(
    model="cross-encoder/ms-marco-TinyBERT-L-6-v2",
    device="cpu"  # or "cuda" for GPU acceleration
)

# Choose based on your needs
memory = Memuri(reranker=balanced_reranker)
```

## Asynchronous Reranking for UI Responsiveness

For user interfaces, you can implement progressive result loading:

```python
import asyncio
from memuri import Memuri

memory = Memuri()
await memory.initialize()

async def progressive_search(query: str):
    # Get initial results quickly
    initial_results = await memory.search_memory(
        query=query,
        rerank=False,  # No reranking for speed
        limit=10
    )
    
    # Return initial results immediately
    yield {"status": "initial", "results": initial_results}
    
    # Start reranking in background
    reranked_results_future = asyncio.create_task(
        memory.search_memory(
            query=query,
            rerank=True,  # With reranking
            limit=10
        )
    )
    
    # Wait for reranked results
    reranked_results = await reranked_results_future
    
    # Return improved results when ready
    yield {"status": "reranked", "results": reranked_results}

# Usage in an async context
async def search_and_update_ui(query):
    async for result_batch in progressive_search(query):
        if result_batch["status"] == "initial":
            # Update UI with initial results
            display_results(result_batch["results"], "Initial results:")
        else:
            # Update UI with improved results
            display_results(result_batch["results"], "Improved results:")
```

## Reranking with Multiple Queries

For complex information needs, combine multiple query aspects:

```python
from memuri import Memuri

memory = Memuri()
await memory.initialize()

# Search with multi-aspect query
results = await memory.search_memory_multi(
    queries=[
        "project deadline", 
        "quarterly report", 
        "budget constraints"
    ],
    weights=[0.5, 0.3, 0.2],  # Weight of each aspect
    limit=7
)
```

## Debugging and Monitoring Reranking

To understand how reranking is affecting results:

```python
from memuri import Memuri
from memuri.core.config import RerankerConfig
import logging

# Enable debug mode for reranker
reranker_config = RerankerConfig(
    debug=True,  # Enables detailed scoring info
    log_level=logging.DEBUG
)

memory = Memuri(reranker=reranker_config)
await memory.initialize()

# Search with debug info
results = await memory.search_memory(
    query="Important project information",
    explain=True  # Returns explanation with scores
)

# Display detailed scoring
for doc, score, explanation in results:
    print(f"Document: {doc.content[:50]}...")
    print(f"Final score: {score:.4f}")
    print(f"Vector score: {explanation['vector_score']:.4f}")
    print(f"Cross-encoder: {explanation['cross_encoder_score']:.4f}")
    print(f"Time decay: {explanation['time_decay_score']:.4f}")
    print(f"Metadata: {explanation['metadata_score']:.4f}")
    print("---")
```

This cookbook demonstrates how to leverage Memuri's reranking capabilities to significantly improve memory retrieval quality while maintaining high performance. 