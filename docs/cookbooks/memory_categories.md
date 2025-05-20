# Memory Categories Cookbook

This cookbook explains how to effectively use Memuri's memory categorization system to organize and manage conversational memories.

## Understanding Memory Categories

Memuri uses a category system to organize memories by their semantic purpose. Default categories include:

- `PERSONAL`: Personal details, preferences, biographical information
- `TASK`: Action items, todos, commitments
- `QUESTION`: Queries requiring answers
- `EMOTION`: Emotional reactions, sentiment 
- `DECISION`: Choices, determinations, conclusions
- `TODO`: Action items specifically marked as todos

## Using the Category System

### 1. Manual Categorization

You can explicitly assign categories when adding memories:

```python
from memuri import Memuri

memory = Memuri()
await memory.initialize()

# Add with explicit category
await memory.add_memory(
    content="Send the quarterly report by Friday",
    category="TASK"
)
```

### 2. Automatic Categorization

Without a specified category, Memuri's classifier will assign one automatically:

```python
# Will be auto-classified by content
await memory.add_memory(
    content="Remember to call the dentist tomorrow at 2pm"
)  # Likely classified as TASK or TODO

await memory.add_memory(
    content="John prefers tea over coffee"
)  # Likely classified as PERSONAL
```

### 3. Category-Specific Search

Filter memories by category during search:

```python
# Only retrieve TASK items
task_results = await memory.search_memory(
    query="What do I need to do?",
    category="TASK",
    limit=5
)
```

## Customizing the Category System

### Configure Category Rules

You can customize how different categories are handled via configuration:

```python
from memuri import Memuri
from memuri.core.config import MemoryRules

# Custom rules for memory categories
custom_rules = MemoryRules(
    TASK={"threshold": 0.75, "action": "add"},      # Auto-add to long-term if confidence >= 0.75
    QUESTION={"action": "none"},                    # Don't store questions
    EMOTION={"action": "short_term", "ttl": 3600},  # Only cache in-memory for 1 hour
    TODO={"action": "add", "priority": "high"}      # Add to long-term with high priority
)

# Initialize with custom rules
memory = Memuri(memory_rules=custom_rules)
```

### Custom Categories

You can extend the category system with your own categories:

```python
from memuri import Memuri
from memuri.core.config import MemoryRules

# Define custom categories and rules
custom_rules = MemoryRules(
    # Standard categories
    TASK={"threshold": 0.75, "action": "add"},
    
    # Custom categories
    PROJECT={"threshold": 0.6, "action": "add"},
    CLIENT={"threshold": 0.8, "action": "add", "ttl": 7776000},  # 90 days retention
    REFERENCE={"action": "add", "index": "exact_match"}          # Use exact matching for references
)

memory = Memuri(memory_rules=custom_rules)
```

## Training the Classifier

Memuri's category classifier improves over time through feedback:

```python
# Explicit feedback to improve classification
await memory.log_feedback(
    content="The API keys expire next month",
    category="TASK"  # This helps train the classifier for similar content
)

# Alternatively, use the log_classification method
await memory.log_classification(
    memory_id="mem_12345",
    correct_category="PROJECT"  # Corrects a misclassified item
)
```

## Hot Words and Trigger Phrases

Configure "hot words" that trigger specific categorization:

```python
from memuri import Memuri
from memuri.core.config import HotwordTriggers

# Define hot words that trigger specific categorization
hot_words = HotwordTriggers(
    TODO=["remember", "don't forget", "todo", "to-do", "need to"],
    TASK=["must", "should", "have to", "deadline"],
    PERSONAL=["likes", "prefers", "favorite", "birthday", "anniversary"]
)

memory = Memuri(hotword_triggers=hot_words)
```

## Batch Classification

For efficiency, classify multiple items at once:

```python
# Classify a batch of items
items = [
    "Send the report by Friday",
    "John's favorite color is blue",
    "What is the capital of France?",
    "Remember to call mom tonight"
]

categories = await memory.classify_batch(items)
for item, category in zip(items, categories):
    print(f"{category}: {item}")
```

## Advanced: Custom Classification Model

For specialized applications, you can provide your own classification model:

```python
from memuri import Memuri
from memuri.services.classifier import CustomClassifier

# Implement your custom classifier
class MyClassifier(CustomClassifier):
    async def initialize(self):
        # Load your model
        pass
        
    async def classify(self, text: str) -> dict:
        # Your classification logic
        # Return format: {"category": "CATEGORY_NAME", "confidence": 0.95}
        pass
        
    async def train(self, texts: list, categories: list):
        # Training logic
        pass

# Use custom classifier
memory = Memuri(classifier=MyClassifier())
```

## Lifecycle Management by Category

Different retention policies can be applied by category:

```python
from memuri import Memuri
from memuri.core.config import RetentionPolicy

# Define retention by category
retention = RetentionPolicy(
    PERSONAL={"days": 365},       # Keep personal info for a year
    TASK={"days": 30},            # Keep tasks for a month
    EMOTION={"days": 7},          # Keep emotional reactions for a week
    TODO={"count": 100},          # Keep only latest 100 todos
    DEFAULT={"days": 90}          # Default retention policy
)

memory = Memuri(retention_policy=retention)

# Apply retention policies (typically scheduled)
await memory.apply_retention_policy()
```

This cookbook demonstrates how to leverage Memuri's category system to create more structured, intelligent memory management in your applications. 