# Memuri - High-Performance Memory SDK

Memuri is a pip-installable SDK for high-performance, pluggable conversational memory services. Designed for local, self-hosted deployment with sub-second latency and production-grade practices.

## Key Features

- **Self-hosted**: No external dependencies or licenses required
- **High Performance**: Sub-100ms memory operations with Faiss and Redis caching
- **Pluggable Architecture**: Swap vector databases, task queues, and LLMs without code changes
- **Production-grade**: Integrated OpenTelemetry traces, health checks, and robust CI/CD
- **Memory Categories**: Organize context by category (PERSONAL, TASK, QUESTION, etc.)
- **Feedback Loop**: Adaptive classification based on user feedback
- **Reranking**: Advanced cross-encoder reranking for more relevant results
- **Flexible Configuration**: Easy configuration with environment variables, dictionary-based config, or direct settings
- **Multiple Embedding Providers**: Support for OpenAI, Google Gemini, Azure, and Sentence Transformers

## Installation

You can install Memuri directly from PyPI:

```bash
pip install memuri
```

Or using Poetry:

```bash
poetry add memuri
```

## Getting Started

### Basic Usage

```python
from memuri import Memuri

# Initialize memory with default settings
memory = Memuri()

# Add a memory item
memory.add_memory(content="John's favorite color is blue", category="PERSONAL")

# Search for relevant memories
results = memory.search_memory("What does John like?")
print(results)
```

### Using Config Dictionary

```python
import os
from memuri import Memuri

# Set API key in environment variable
os.environ["OPENAI_API_KEY"] = "your_api_key"

# Create config with specific provider settings
config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
}

# Initialize with config
memory = Memuri.from_config(config)
```

### Storing Chat Conversations

```python
# Add a conversation as memory
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about sci-fi? I recommend Interstellar."},
    {"role": "user", "content": "I love sci-fi movies!"}
]

# Store the conversation with user ID
await memory.add(messages, user_id="john")
```

## Architecture

Memuri is designed around a layered memory system:

1. **Short-Term Memory**: HNSW In-Memory Index + Redis LRU Cache
2. **Long-Term Memory**: Pluggable Vector Stores (pgvector, Milvus, Qdrant, Redis Vector)
3. **Memory Triggers**: Category classifiers and rule engine for contextual decisions
4. **Feedback Loop**: Continuous adaptation based on user interactions

## Performance Testing

Memuri includes a comprehensive latency testing framework to measure performance metrics:

### Using the Latency Test Script

```bash
# Run all tests with default settings
cd src/memuri/tests
./run_latency_tests.py

# Run only add memory tests with 20 iterations
./run_latency_tests.py --test-type add --iterations 20

# Run only search tests with custom database URL
./run_latency_tests.py --test-type search --db-url postgresql://user:pass@localhost:5432/mydb

# Set log level to DEBUG for more detailed output
./run_latency_tests.py --log-level DEBUG

# Don't save results to a file
./run_latency_tests.py --no-save
```

### Test Output

The test runner will output detailed statistics for each operation:

```
=== Add Memory Benchmark Results ===
Total operations: 10
Mean latency: 831.96ms
Median latency: 990.54ms
Min latency: 322.88ms
Max latency: 1361.35ms
95th percentile: 1273.73ms
Standard deviation: 387.65ms
==================================

=== Search Memory Benchmark Results ===
Total operations: 10
Mean latency: 817.86ms
Median latency: 804.20ms
Min latency: 410.45ms
Max latency: 1388.41ms
95th percentile: 1283.28ms
Standard deviation: 359.60ms
==================================

```

### Programmatic Usage

You can also use the LatencyBenchmark class in your own code:

```python
import asyncio
from memuri import Memuri
from memuri.tests.test_latency import LatencyBenchmark

async def measure_performance():
    client = Memuri()
    benchmark = LatencyBenchmark("Custom Test")
    
    # Measure add_memory operation
    for i in range(5):
        await benchmark.measure_operation(
            client.add_memory,
            content=f"Test memory {i}",
            category="FACT"
        )
    
    # Print stats
    benchmark.print_stats()
    
    # Get stats as dictionary
    stats = benchmark.get_stats()
    print(f"95th percentile latency: {stats['p95']:.2f}ms")

asyncio.run(measure_performance())
```

## Development and Deployment

### Publishing to PyPI

The project includes a script to handle publishing to PyPI with automatic version management:

```bash
# From project root
./scripts/upload_to_pypi.sh
```

This script will:
1. Automatically increment version numbers (patch, minor, or major)
2. Run tests before building
3. Build and upload the package to PyPI
4. Create git tags and commits for the release

## Next Steps

- Check the [Quick Start](usage/quickstart.md) guide
- Learn about [Configuration](usage/configuration.md) options
- Explore [API Reference](api-reference/index.md) 
- Learn advanced patterns with [Cookbooks](cookbooks/index.md)
- View [Examples](examples/index.md) for complete solutions

## Updates and Improvements

### Latest Updates (v0.2.0)

We've enhanced Memuri with several key improvements:

1. **Enhanced Category System**
   - Hierarchical memory categorization with main categories and subcategories
   - 15 main categories and 45+ subcategories for precise memory organization
   - Backward compatibility with legacy categories

2. **Improved OpenAI Embedding Support**
   - Dynamic client configuration with proper error handling
   - Support for all OpenAI embedding models including text-embedding-3-small/large
   - Advanced configuration options (proxies, Azure integration, custom parameters)

3. **Feedback System Enhancement**
   - Track and analyze feedback for both categories and subcategories
   - Improved classifier training based on user feedback
   - Parent-child relationship awareness in memory classification

4. **Configuration Flexibility**
   - Enhanced `from_config` method for cleaner, more intuitive setup
   - Direct parameter passing to embedding and LLM services
   - Better factory patterns for service initialization

Check the documentation for details on how to use these new features.