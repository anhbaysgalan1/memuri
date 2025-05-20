# Memuri Cookbooks

These cookbooks provide detailed guidance for specific Memuri use cases and features. Each recipe offers a working example with explanations to help you integrate Memuri into your applications.

## Available Cookbooks

- [Memory Categories](memory_categories.md) - Learn how to use and customize the memory categorization system
- [Reranking Pipeline](reranking_pipeline.md) - Implement advanced reranking for more relevant search results
- [Custom Vector Stores](custom_vectorstores.md) - Connect to different vector databases
- [Feedback Loop](feedback_loop.md) - Implement a feedback-driven memory system
- [Memory Lifecycle](memory_lifecycle.md) - Manage memory retention and cleanup
- [Performance Tuning](performance_tuning.md) - Optimize for high-throughput or low-latency use cases
- [Embedder Customization](embedder_customization.md) - Configure and customize embedding models

## Concepts

Memuri is built around several key concepts:

### Memory Tiers

Memuri uses a tiered memory approach:

1. **Short-term memory** (HNSW in-memory index + Redis cache)
2. **Long-term memory** (Vector database)

The tiers work together to provide sub-100ms lookup times while maintaining persistence and scale.

### Memory Categories

Memory items can be categorized (like `PERSONAL`, `TASK`, `QUESTION`, etc.) for more targeted retrieval and automatic lifecycle management. The category system can be used to:

- Filter search results
- Apply different storage policies
- Define automated cleanup rules

### Feedback Loop

Memuri includes a feedback loop mechanism that:

1. Captures user interactions with memory items
2. Refines classifiers based on feedback
3. Adapts memory triggering rules over time

This creates a self-improving memory system that gets better with use.

## Complementary Documentation

For a complete understanding of Memuri, also check:

- [API Reference](../api-reference/index.md) - Detailed technical documentation
- [Examples](../examples/index.md) - Complete working examples
- [Usage Guides](../usage/index.md) - Step-by-step guides for common tasks 