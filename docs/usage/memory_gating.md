# Memory Gating Layer

## Overview

The Memory Gating layer in Memuri filters incoming information before storage in long-term memory. It intelligently decides what's worth remembering based on several criteria:

- **Basic rules**: Minimum length, skip words
- **Similarity checks**: Avoid redundancy with recent memories  
- **Relevance classification**: ML-based importance prediction

## Key Benefits

- Reduces noise in your memory system
- Prevents information duplication
- Prioritizes important information
- Runs locally for fast (<10ms) decisions

## Usage

```python
from memuri.factory import GatingFactory, EmbedderFactory, ClassifierFactory

# Create dependencies
embedding_service = EmbedderFactory.create()
classifier_service = ClassifierFactory.create()

# Create memory gate with custom parameters
memory_gate = GatingFactory.create(
    embedding_service=embedding_service,
    classifier_service=classifier_service,
    # Optional customization
    similarity_threshold=0.85,  # Higher = stricter filtering
    min_content_length=30,      # Minimum text length to store
    keep_phrases=["remember", "important"],  # Always keep these
)

# Evaluate text
should_store, reason = await memory_gate.evaluate(
    "Some text to evaluate"
)

# Store if it passes
if should_store:
    # Store in your memory system
```

## Integration with MemoryOrchestrator

The MemoryOrchestrator includes memory gating by default:

```python
from memuri.services.memory import MemoryOrchestrator

# Create orchestrator with services
orchestrator = MemoryOrchestrator(
    memory_service=memory_service,
    embedding_service=embedding_service,
    classifier_service=classifier_service,
    # Other services...
)

# Add with gating
stored, reason, memory = await orchestrator.add_memory(
    content="Text to evaluate and potentially store",
    # Skip gating with skip_gating=True if needed
)
```

## Examples

See complete examples:
- [Basic example](../examples/memory_gating.py)  
- [Chatbot with memory gating](../examples/chatbot_with_gating.py)
- [Detailed documentation](memory_gating.md)

## Tuning Guidelines

- Start with permissive settings, gradually tighten
- Monitor acceptance rates
- Use explicit keep phrases for important info
- Customize parameters based on your use case:
  - Shorter `min_content_length` for chat (15-20 chars)
  - Higher `similarity_threshold` (>0.9) for less duplicate filtering
  - Lower `confidence_threshold` (<0.3) if using ML classifier 