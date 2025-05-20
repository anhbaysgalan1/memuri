# Memuri Quickstart Guide

This guide will help you quickly get up and running with Memuri, the high-performance memory layer for conversational AI.

## Installation

```bash
pip install memuri
```

## Prerequisites

Memuri requires:

- Python 3.11+
- A PostgreSQL database with pgvector extension installed (for vector storage)
- Redis (for caching and task queue)

You can use the included Docker Compose file to quickly set up the required services:

```bash
# From the project root
docker-compose up -d
```

## Basic Setup

```python
from memuri import Memuri

# Initialize with default settings
memory = Memuri()

# Or customize connections
memory = Memuri(
    vector_store="pgvector",  # Options: pgvector, milvus, qdrant, redis_vector
    vector_store_connection="postgresql://postgres:postgres@localhost:5432/memuri",
    cache_connection="redis://localhost:6379/0",
    embedder="openai",  # Options: openai, google, sentence_transformers
    embedder_model="text-embedding-ada-002",  # For OpenAI
)

# Connect to services
await memory.initialize()
```

## Configuration-Based Setup

For more flexible configuration, you can use the `from_config` method:

```python
import os
from memuri import Memuri

# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = "your-api-key" 

# Create configuration dictionary
config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002"
        }
    },
    "vector_store": {
        "provider": "pgvector",
        "connection_string": "postgresql://postgres:postgres@localhost:5432/memuri"
    }
}

# Create Memuri instance from config
memory = Memuri.from_config(config)
await memory.initialize()
```

## Adding Memuries

You can add memories with or without explicit categories:

```python
# Add with explicit category
await memory.add_memory(
    content="John's favorite color is blue",
    category="PERSONAL",
    metadata={"source": "conversation", "timestamp": "2023-06-15T10:30:00Z"}
)

# Add without category (will be auto-classified)
await memory.add_memory(
    content="Remember to send the project proposal by Friday",
)

# Add multiple memories in batch
await memory.add_memories([
    {"content": "Sarah's birthday is on March 15", "category": "PERSONAL"},
    {"content": "The meeting is scheduled for 2PM tomorrow", "category": "TASK"}
])
```

## Adding Chat Conversations

You can add entire chat conversations to memory:

```python
# Define a conversation
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movie? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll recommend sci-fi movies in the future."}
]

# Store the entire conversation with user ID
await memory.add(messages, user_id="john")

# Later, search for movie preferences
results = await memory.search_memory("What kind of movies does John like?")
```

## Searching Memuries

Retrieve relevant memories based on a query:

```python
# Basic search (returns all relevant memories)
results = await memory.search_memory("What does John like?")

# Search with category filter
results = await memory.search_memory(
    query="What are my tasks?",
    category="TASK",
    limit=5
)

# Display results
for doc, score in results:
    print(f"[{score:.2f}] {doc.content}")
```

## Using Memory with Chat

Memuri is designed to augment conversational AI by providing relevant context:

```python
from memuri import Memuri
import openai

memory = Memuri()
await memory.initialize()

# In your chat loop
async def chat_with_memory(user_message):
    # Add user message to memory if appropriate
    if "remember" in user_message.lower():
        await memory.add_memory(content=user_message)
    
    # Retrieve relevant context from memory
    memories = await memory.search_memory(user_message, limit=3)
    
    # Format memories as context
    context = ""
    if memories:
        context = "Relevant information:\n" + "\n".join([
            f"- {item[0].content}" for item in memories
        ])
    
    # Call your LLM with context
    completion = await openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. " + context},
            {"role": "user", "content": user_message}
        ]
    )
    
    return completion.choices[0].message.content

# Example usage
response = await chat_with_memory("What is John's favorite color?")
print(response)
```

## Google Gemini Integration

If you prefer using Google's Gemini embeddings:

```python
import os
from memuri import Memuri

# Set Google API key
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"

# Configure with Google embeddings
config = {
    "embedder": {
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "model_kwargs": {
                "task_type": "retrieval_document"
            }
        }
    }
}

memory = Memuri.from_config(config)
await memory.initialize()

# Now use memory as normal
await memory.add_memory("Important meeting tomorrow at 2pm", category="TASK")
results = await memory.search_memory("What's on my schedule?")
```

## Next Steps

- Check out our [Cookbooks](../cookbooks/index.md) for advanced use cases
- Learn about the [Memory Categories](../cookbooks/memory_categories.md) system
- Explore [Vector Store Adapters](../api-reference/adapters.md) for different backends
- See [Complete Examples](../examples/index.md) for integrations 
- Read about [Configuration Options](configuration.md) for customizing Memuri 