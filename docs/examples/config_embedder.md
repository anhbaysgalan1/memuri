# Configuring Embedders in Memuri

This example demonstrates how to configure different embedding providers using the config-based approach in Memuri.

## Basic Config with OpenAI

```python
import os
from memuri import Memuri

# Set API key in environment variable
os.environ["OPENAI_API_KEY"] = "your_api_key"

# Create config with OpenAI embedder
config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002",
        }
    }
}

# Create Memuri instance from config
memory = Memuri.from_config(config)

# Add a memory
await memory.add_memory("John likes blue color", category="PERSONAL")

# Search using the configured embedder
results = await memory.search_memory("What does John like?")
```

## Using Google's Gemini Embeddings

```python
import os
from memuri import Memuri

# Set API key in environment variable
os.environ["GOOGLE_API_KEY"] = "your_google_api_key"

# Create config with Google Gemini embedder
config = {
    "embedder": {
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "model_kwargs": {
                "task_type": "retrieval_document"  # or "retrieval_query", "semantic_similarity", etc.
            }
        }
    }
}

# Create Memuri instance from config
memory = Memuri.from_config(config)

# Add a memory
await memory.add_memory("The meeting is scheduled for tomorrow at 3 PM", category="TASK")
```

### Advanced Google Gemini Configuration

```python
from memuri import Memuri

# Create a more customized Google embedder configuration
config = {
    "embedder": {
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": "your_google_api_key",
            "embedding_dims": 768,
            "batch_size": 20,
            "model_kwargs": {
                "task_type": "semantic_similarity"
            }
        }
    }
}

memory = Memuri.from_config(config)
```

## Using Explicit API Keys in Config

You can also provide API keys directly in the config instead of using environment variables:

```python
from memuri import Memuri

# Create config with explicit API key
config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": "your_openai_api_key",
            "embedding_dims": 1536
        }
    }
}

# Create Memuri instance from config
memory = Memuri.from_config(config)
```

## Using Local Sentence Transformers

For privacy-focused or offline use cases, you can use sentence-transformers:

```python
from memuri import Memuri

# Create config with sentence_transformers embedder
config = {
    "embedder": {
        "provider": "sentence_transformers",
        "config": {
            "model": "all-MiniLM-L6-v2",
            "model_kwargs": {
                "device": "cuda"  # Use GPU if available
            }
        }
    }
}

# Create Memuri instance from config
memory = Memuri.from_config(config)
```

## Using Azure OpenAI

```python
from memuri import Memuri

# Create config with Azure OpenAI
config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "your-deployment-name",
            "api_key": "your_azure_openai_key",
            "azure_kwargs": {
                "api_version": "2023-05-15",
                "azure_endpoint": "https://your-resource-name.openai.azure.com"
            }
        }
    }
}

# Create Memuri instance from config
memory = Memuri.from_config(config)
```

## Adding Chat Messages

You can add chat conversations to memory:

```python
import os
from memuri import Memuri

os.environ["OPENAI_API_KEY"] = "your_api_key"

config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002",
        }
    }
}

memory = Memuri.from_config(config)

# Add a conversation as memory
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movie? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]

# Store the conversation with a user ID
await memory.add(messages, user_id="john")

# Later, search for relevant memories
results = await memory.search_memory("What kind of movies does John like?")
for memory, score in results:
    print(f"[{score:.2f}] {memory.content}")
```

## Configuring Multiple Services Together

You can configure multiple services at once for a complete setup:

```python
from memuri import Memuri

config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002",
            "api_key": "your_openai_api_key"
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4",
            "api_key": "your_openai_api_key",
            "temperature": 0.7
        }
    }
}

memory = Memuri.from_config(config)

# Add memories and interact with the LLM
await memory.add_memory("Remember that Julia prefers vegetarian food", category="PERSONAL")

# Using both services: LLM access for response generation, memory for context retrieval
context = await memory.search_memory("What does Julia like to eat?")
response = await memory.generate_text(
    f"Context: {context[0][0].content if context else 'No information found'}\n"
    "Question: What should I cook for Julia?"
)
print(response)
```

## Full Configuration Options

Here's a comprehensive example showing many configuration options:

```python
from memuri import Memuri

config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002",
            "api_key": "your_api_key",
            "base_url": "https://api.openai.com/v1",  # Custom base URL if needed
            "embedding_dims": 1536,
            "http_client_proxies": {  # Optional proxy settings
                "http": "http://proxy.example.com:8080",
                "https": "https://proxy.example.com:8080"
            }
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4",
            "api_key": "your_api_key",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    },
    "vector_store": {
        "provider": "pgvector",
        "connection_string": "postgresql://postgres:postgres@localhost:5432/memuri",
        "dimension": 1536
    },
    "cache": {
        "provider": "redis",
        "connection_string": "redis://localhost:6379/0",
        "ttl": 3600
    }
}

memory = Memuri.from_config(config)
```

This flexible configuration system allows you to easily switch between different embedding providers and customize their behavior without changing your code. 