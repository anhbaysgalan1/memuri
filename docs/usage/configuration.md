# Configuring Memuri

This guide explains how to configure Memuri for different environments and use cases.

## Configuration Methods

Memuri provides multiple ways to configure the SDK:

1. **Environment variables** - Set configurations through environment variables
2. **Config dictionary** - Use a Python dictionary with the `from_config` method
3. **Direct parameters** - Pass settings directly to the `Memuri` constructor
4. **.env file** - Use a .env file for environment-specific configuration

## Environment Variables

The simplest way to configure Memuri is through environment variables. Memuri looks for variables with the `MEMURI_` prefix:

```python
import os
from memuri import Memuri

# Set required environment variables
os.environ["MEMURI_DATABASE__POSTGRES_URL"] = "postgresql://postgres:postgres@localhost:5432/memuri"
os.environ["MEMURI_REDIS__REDIS_URL"] = "redis://localhost:6379/0"
os.environ["OPENAI_API_KEY"] = "sk-your-api-key"

# Create Memuri instance (will use environment variables)
memory = Memuri()
```

## Using the from_config Method

For more flexibility, you can use the `from_config` method with a configuration dictionary:

```python
from memuri import Memuri

config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": "sk-your-api-key"
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4",
            "temperature": 0.7
        }
    }
}

memory = Memuri.from_config(config)
```

## API Key Configuration

Memuri supports multiple ways to specify API keys for different services:

### 1. Through Standard Environment Variables

Memuri automatically recognizes standard environment variables from providers:

```python
import os
from memuri import Memuri

# Set standard API keys as environment variables
os.environ["OPENAI_API_KEY"] = "sk-your-openai-key"
os.environ["GOOGLE_API_KEY"] = "your-google-key"

# Memuri will automatically use these keys
memory = Memuri()
```

### 2. Through Memuri-specific Environment Variables

You can also use Memuri-specific variables:

```python
import os
from memuri import Memuri

# Set Memuri-specific API key environment variables
os.environ["MEMURI_EMBEDDING__API_KEY"] = "sk-your-openai-key" 
os.environ["MEMURI_EMBEDDING__PROVIDER"] = "openai"
os.environ["MEMURI_EMBEDDING__MODEL_NAME"] = "text-embedding-3-small"

memory = Memuri()
```

### 3. Through Configuration Dictionary

The most flexible approach is using the config dictionary:

```python
from memuri import Memuri

config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": "sk-your-openai-key",
            "base_url": "https://api.openai.com/v1"  # Optional custom base URL
        }
    }
}

memory = Memuri.from_config(config)
```

For Google Gemini:

```python
config = {
    "embedder": {
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": "your-google-key",
            "model_kwargs": {
                "task_type": "retrieval_document"  # Set task type for embedding
            }
        }
    }
}
```

### 4. Through Settings Objects

You can also create settings objects directly:

```python
from memuri import Memuri, EmbeddingSettings, MemuriSettings

# Create custom settings
embedding_settings = EmbeddingSettings(
    provider="openai",
    model_name="text-embedding-3-small",
    api_key="sk-your-openai-key"
)

settings = MemuriSettings(
    embedding=embedding_settings
)

# Initialize with custom settings
memory = Memuri(settings=settings)
```

## Embedder Configuration

Configure embedding providers with various options:

### OpenAI

```python
config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",  # or "text-embedding-3-small", "text-embedding-3-large"
            "api_key": "sk-your-openai-key",
            "embedding_dims": 1536,
            "batch_size": 32  # Process in batches for better performance
        }
    }
}
```

### Google (Gemini)

```python
config = {
    "embedder": {
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": "your-google-key",
            "embedding_dims": 768,
            "model_kwargs": {
                "task_type": "retrieval_document"  # Type options: "retrieval_document", "retrieval_query", "semantic_similarity", "classification", "clustering"
            }
        }
    }
}
```

### Sentence Transformers (Local)

```python
config = {
    "embedder": {
        "provider": "sentence_transformers",
        "config": {
            "model": "all-MiniLM-L6-v2",
            "model_kwargs": {
                "device": "cuda"  # or "cpu"
            }
        }
    }
}
```

### Azure OpenAI

```python
config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "deployment-name",
            "api_key": "your-azure-key",
            "azure_kwargs": {
                "api_version": "2023-05-15",
                "azure_endpoint": "https://your-resource.openai.azure.com"
            }
        }
    }
}
```

## Vector Store Configuration

Configure different vector stores:

### pgvector (PostgreSQL)

```python
config = {
    "vector_store": {
        "provider": "pgvector",
        "connection_string": "postgresql://postgres:postgres@localhost:5432/memuri",
        "dimension": 1536,
        "index_type": "hnsw"
    }
}
```

### Redis Vector

```python
config = {
    "vector_store": {
        "provider": "redis_vector",
        "connection_string": "redis://localhost:6379/0",
        "dimension": 1536,
        "distance_metric": "COSINE"
    }
}
```

## Full Configuration Example

Here's a comprehensive configuration example:

```python
config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": "sk-your-openai-key",
            "batch_size": 32
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4",
            "api_key": "sk-your-openai-key",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    },
    "vector_store": {
        "provider": "pgvector",
        "connection_string": "postgresql://postgres:postgres@localhost:5432/memuri"
    },
    "cache": {
        "provider": "redis",
        "connection_string": "redis://localhost:6379/0",
        "ttl": 3600
    }
}

memory = Memuri.from_config(config)
```

## Chat Message Storage

Memuri now supports adding chat messages in a format compatible with other memory systems:

```python
from memuri import Memuri

memory = Memuri.from_config(config)

# Add a conversation as memory
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movie? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."}
]

# Store with additional metadata
await memory.add(messages, user_id="john", metadata={"session_id": "abc123"})
```

## Advanced Settings

For advanced use cases, you can configure additional aspects of Memuri:

### Memory Rules

```python
from memuri import Memuri
from memuri.core.config import MemoryRuleSettings

# Create custom memory rules
memory_rules = {
    "TASK": MemoryRuleSettings(threshold=0.7, action="add"),
    "QUESTION": MemoryRuleSettings(action="none"),
    "PERSONAL": MemoryRuleSettings(threshold=0.8, action="add")
}

# Initialize with custom rules
memory = Memuri(settings=MemuriSettings(memory_rules=memory_rules))
```

### Reranker Configuration

```python
from memuri import Memuri
from memuri.core.config import RerankSettings

# Configure reranker
rerank_settings = RerankSettings(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    alpha=0.7,  # Cross-encoder score weight
    beta=0.2,   # Time decay weight
    gamma=0.1   # Metadata score weight
)

# Initialize with custom reranker
memory = Memuri(settings=MemuriSettings(rerank=rerank_settings))
```

## Environment-Specific Configuration

Use a `.env` file for environment-specific configuration:

```
# .env file
MEMURI_DATABASE__POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/memuri
MEMURI_REDIS__REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=sk-your-api-key
MEMURI_EMBEDDING__PROVIDER=openai
MEMURI_EMBEDDING__MODEL_NAME=text-embedding-3-small
```

Then simply:

```python
from memuri import Memuri

# Will read from .env file
memory = Memuri()
``` 