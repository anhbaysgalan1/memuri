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

## Getting Started

```bash
pip install memuri
```

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
            "model": "text-embedding-ada-002"
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

## Next Steps

- Check the [Quick Start](usage/quickstart.md) guide
- Learn about [Configuration](usage/configuration.md) options
- Explore [API Reference](api-reference/index.md) 
- Learn advanced patterns with [Cookbooks](cookbooks/index.md)
- View [Examples](examples/index.md) for complete solutions 