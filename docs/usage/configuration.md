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

## Memory Categories

Memuri now supports a rich hierarchical category system with main categories and subcategories:

```python
# Example memory categories
from memuri.domain.models import MemoryCategory

# Main categories
profile = MemoryCategory.PROFILE_INFORMATION
tasks = MemoryCategory.PROJECTS_TASKS

# Subcategories
personal_info = MemoryCategory.PERSONAL_INFORMATION
todo_items = MemoryCategory.TO_DO_ITEMS
```

The full category hierarchy includes:

1. **Profile Information**
   - Personal Information
   - Demographics
   - Identity Traits
   - Personal Details

2. **Preferences**
   - Favorite Topics
   - Communication Style
   - Media Preferences

3. **Goals & Aspirations**
   - Career Goals
   - Personal Goals
   - Project Aspirations

4. **Routines & Habits**
   - Daily Routines
   - Health Habits
   - Productivity Habits

5. **Events & Appointments**
   - Calendar Events
   - Milestones
   - Travel Plans

6. **Projects & Tasks**
   - Active Projects
   - To-Do Items
   - Backlog Items

7. **Health & Wellness**
   - Medical Conditions
   - Dietary Preferences
   - Wellness Metrics

8. **Social Relationships**
   - Family Members
   - Friends Network
   - Professional Contacts

9. **Skills & Knowledge**
   - Technical Skills
   - Languages Spoken
   - Certifications

10. **Experiences & Memories**
    - Travel Experiences
    - Educational Background
    - Notable Life Events

11. **Feedback & Opinions**
    - Product Feedback
    - Personal Opinions
    - Suggestions

12. **Financial Info**
    - Budget Goals
    - Expenses Log
    - Investment Preferences

13. **Media Content**
    - Books Read
    - Articles Consumed 
    - Multimedia Engagement

14. **Contextual Metadata**
    - Device Info
    - Session Preferences
    - Location History

15. **Miscellaneous**
    - Misc

## Enhanced OpenAI Embedding Configuration

The OpenAI embedding service supports advanced configurations:

```python
config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",  # or "text-embedding-3-large", "text-embedding-ada-002"
            "api_key": "sk-your-openai-key",    # Optional, can also use OPENAI_API_KEY env var
            "embedding_dims": 1536,             # 1536 for text-embedding-3-small, 3072 for text-embedding-3-large
            "batch_size": 32,                   # Process in batches for better performance
            # Advanced configuration options
            "base_url": "https://api.openai.com/v1",  # Optional, for custom endpoints
            "http_client_proxies": {                  # Optional, for proxy configuration
                "http": "http://proxy.example.com:8080",
                "https": "https://proxy.example.com:8080"
            },
            "model_kwargs": {                        # Optional, additional model parameters
                "encoding_format": "float"           # Optional, "float" (default) or "base64"
            }
        }
    }
}
```

## Feedback System

Memuri includes a feedback system for improving memory classification over time:

```python
from memuri import Memuri
from memuri.domain.models import MemoryCategory

# Initialize client
memuri = Memuri()

# Add a memory that was auto-classified
memory = await memuri.add_memory(
    content="I need to buy milk today"
)

# If the classification was incorrect, provide feedback
await memuri.log_feedback(
    text="I need to buy milk today",
    category=MemoryCategory.TO_DO_ITEMS,  # Correct category
    metadata={"user_id": "1234"}  # Optional additional info
)
``` 