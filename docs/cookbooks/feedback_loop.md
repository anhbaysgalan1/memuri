# Feedback Loop Cookbook

This cookbook demonstrates how to implement a feedback-driven memory system with Memuri that continuously improves based on user interactions.

## Understanding the Feedback Loop

Memuri's feedback loop enables:

1. **Capturing feedback** on memory categories and relevance
2. **Refining classifiers** based on this feedback
3. **Adapting memory rules** to improve future storage and retrieval

The feedback system helps Memuri learn from user interactions and become more tailored to specific use cases over time.

## Setting Up Feedback Collection

### Basic Feedback Collection

```python
from memuri import Memuri

memory = Memuri()
await memory.initialize()

# Log explicit feedback for classifier improvement
await memory.log_feedback(
    content="Remember to submit the expense report by Friday",
    category="TASK"  # User-defined category
)

# Log feedback about an existing memory
await memory.log_classification_feedback(
    memory_id="mem_12345",
    correct_category="TASK"  # The correct category (if auto-classified incorrectly)
)

# Log relevance feedback
await memory.log_relevance_feedback(
    query="What tasks do I have?",
    memory_id="mem_12345",
    is_relevant=True,  # This memory was relevant to the query
    score=0.9          # Optional relevance score (0.0-1.0)
)
```

### Integrated Feedback Collection in Applications

Here's how to add feedback collection to a chat interface:

```python
from memuri import Memuri

memory = Memuri()
await memory.initialize()

async def handle_chat_message(user_message: str):
    # Process user message
    if user_message.startswith("Remember ") or "note that" in user_message:
        # Store as memory and capture feedback from message intent
        memory_item = await memory.add_memory(content=user_message)
        
        # If user has indicated a category in their message
        if "task:" in user_message.lower():
            await memory.log_classification_feedback(
                memory_id=memory_item["id"],
                correct_category="TASK"
            )
        elif "personal:" in user_message.lower():
            await memory.log_classification_feedback(
                memory_id=memory_item["id"],
                correct_category="PERSONAL"
            )
    
    # Search for relevant memories
    results = await memory.search_memory(user_message, limit=3)
    
    # [... rest of chat processing ...]
    return response
```

## Automated Classifier Retraining

### Scheduled Retraining

```python
import asyncio
from memuri import Memuri

memory = Memuri()
await memory.initialize()

async def scheduled_retraining_job():
    """Job to retrain classifier with accumulated feedback"""
    training_result = await memory.retrain_classifier()
    print(f"Classifier retrained with {training_result['samples']} samples")
    print(f"Categories: {training_result['categories']}")
    print(f"Accuracy: {training_result['accuracy']:.2f}")

# Run retraining every day
async def daily_maintenance():
    while True:
        await scheduled_retraining_job()
        await asyncio.sleep(86400)  # 24 hours

# Start the maintenance loop
asyncio.create_task(daily_maintenance())
```

### Threshold-based Retraining

```python
from memuri import Memuri

memory = Memuri(
    feedback_config={
        "auto_retrain": True,  # Enable automatic retraining
        "min_samples": 50,     # Minimum samples before retraining
        "min_accuracy_gain": 0.02  # Minimum accuracy improvement to apply model
    }
)
await memory.initialize()

# With auto_retrain=True, Memuri will check for sufficient new feedback
# samples after each feedback submission and retrain when appropriate
```

## Implementing Feedback UI Elements

Here's an example of implementing feedback UI in a web application using FastAPI:

```python
from fastapi import FastAPI, HTTPException
from memuri import Memuri
import asyncio

app = FastAPI()
memory = Memuri()

@app.on_event("startup")
async def startup():
    await memory.initialize()

@app.post("/memory/feedback/category")
async def memory_category_feedback(memory_id: str, category: str):
    """Endpoint for users to correct memory categorization"""
    try:
        await memory.log_classification_feedback(
            memory_id=memory_id,
            correct_category=category
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/feedback/relevance")
async def memory_relevance_feedback(query: str, memory_id: str, is_relevant: bool):
    """Endpoint for users to indicate relevance of search results"""
    try:
        await memory.log_relevance_feedback(
            query=query,
            memory_id=memory_id,
            is_relevant=is_relevant
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/admin/retrain")
async def trigger_retraining():
    """Admin endpoint to manually trigger classifier retraining"""
    try:
        result = await memory.retrain_classifier()
        return {
            "status": "success",
            "samples_used": result["samples"],
            "accuracy": result["accuracy"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Advanced Feedback-Driven Tuning

### Adaptive Memory Rules

Automatically adjust memory rules based on feedback patterns:

```python
from memuri import Memuri
from memuri.core.config import AdaptiveRules

memory = Memuri(
    adaptive_rules=AdaptiveRules(
        enabled=True,
        update_interval=7,  # Days between rule updates
        confidence_threshold=0.8,  # Minimum confidence to apply changes
        min_samples_per_rule=25  # Minimum feedback samples to consider
    )
)
await memory.initialize()

# The system will now periodically analyze feedback patterns
# and suggest/apply adjustments to memory rules
```

### Monitoring Feedback Impact

Track how feedback affects system performance:

```python
from memuri import Memuri

memory = Memuri()
await memory.initialize()

# Get feedback system metrics
feedback_metrics = await memory.get_feedback_metrics()

print(f"Total feedback samples: {feedback_metrics['total_samples']}")
print(f"Samples by category: {feedback_metrics['samples_by_category']}")
print(f"Classification accuracy: {feedback_metrics['classification_accuracy']:.2f}")
print(f"Retrieval precision: {feedback_metrics['retrieval_precision']:.2f}")
print(f"Retrieval recall: {feedback_metrics['retrieval_recall']:.2f}")

# Get feedback improvement trend
trend = await memory.get_feedback_improvement_trend(days=30)
print("Accuracy trend over 30 days:", trend["accuracy_trend"])
```

### Custom Feedback Processors

For specialized applications, you can implement custom feedback processors:

```python
from memuri import Memuri
from memuri.services.feedback import FeedbackProcessor

class CustomFeedbackProcessor(FeedbackProcessor):
    """Custom implementation of feedback processing logic"""
    
    async def process_classification_feedback(self, content, category):
        # Custom logic to process classification feedback
        # For example, applying different weights based on source
        await self._log_to_database(content, category)
        
        # Maybe trigger immediate learning for critical categories
        if category in ["CRITICAL", "SECURITY"]:
            await self.apply_immediate_learning(content, category)
            
    async def apply_immediate_learning(self, content, category):
        # Custom logic to immediately apply this feedback
        # without waiting for scheduled retraining
        pass
        
    async def get_training_data(self):
        # Custom logic to prepare training data
        # Maybe combining with external data sources
        pass

# Use custom processor
memory = Memuri(feedback_processor=CustomFeedbackProcessor())
```

### A/B Testing Memory Rules

Experiment with different memory rules to optimize performance:

```python
from memuri import Memuri
from memuri.core.config import MemoryRules, ExperimentConfig

# Define different rule configurations to test
config_a = MemoryRules(
    TASK={"threshold": 0.7, "action": "add"},
    QUESTION={"action": "none"}
)

config_b = MemoryRules(
    TASK={"threshold": 0.6, "action": "add"},
    QUESTION={"action": "short_term", "ttl": 3600}
)

# Setup experiment
experiment = ExperimentConfig(
    name="question_retention_test",
    configs={"A": config_a, "B": config_b},
    traffic_split={"A": 0.5, "B": 0.5},
    duration_days=14,
    success_metric="retrieval_precision"
)

memory = Memuri(experiment=experiment)
await memory.initialize()

# After the experiment completes
results = await memory.get_experiment_results("question_retention_test")
print(f"Winning configuration: {results['winner']}")
print(f"Performance improvement: {results['improvement']:.2f}%")
```

## Feedback Data Export and Analysis

Extract feedback data for external analysis:

```python
from memuri import Memuri
import pandas as pd

memory = Memuri()
await memory.initialize()

# Export feedback data to DataFrame
feedback_data = await memory.export_feedback_data(
    start_date="2023-01-01",
    end_date="2023-06-30"
)

# Convert to pandas DataFrame for analysis
df = pd.DataFrame(feedback_data)

# Analyze category distribution
category_counts = df['category'].value_counts()
print("Feedback by category:", category_counts)

# Analyze accuracy over time
df['date'] = pd.to_datetime(df['timestamp']).dt.date
accuracy_by_date = df.groupby('date')['is_correct'].mean()
print("Accuracy trend:", accuracy_by_date)

# Save for further analysis
df.to_csv("feedback_analysis.csv", index=False)
```

This cookbook demonstrates how to implement a comprehensive feedback loop in Memuri, enabling the system to continuously learn and improve from user interactions. 