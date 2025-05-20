# Integrating Memuri with LLMs

This example demonstrates how to integrate Memuri with different LLM providers to create a chatbot with long-term memory.

## Basic Integration with OpenAI

```python
import asyncio
import os
from memuri import Memuri
from openai import AsyncOpenAI

# Initialize Memuri
memory = Memuri(
    vector_store="pgvector",
    vector_store_connection="postgresql://postgres:postgres@localhost:5432/memuri",
    embedder="openai",
    embedder_model="text-embedding-ada-002"
)
await memory.initialize()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

async def chat_with_memory(user_input: str, chat_history: list):
    """Process a user message with memory-augmented responses."""
    
    # Check if we should remember this message
    should_remember = any(keyword in user_input.lower() for keyword in 
                         ["remember", "note", "don't forget", "keep in mind"])
    
    if should_remember:
        # Store this in memory
        await memory.add_memory(content=user_input)
        print("[Saved to memory]")
    
    # Retrieve relevant memories
    relevant_memories = await memory.search_memory(user_input, limit=3)
    
    # Format memories as context for the LLM
    memory_context = ""
    if relevant_memories:
        memory_context = "Here are some relevant memories I have:\n" + "\n".join([
            f"- {item[0].content}" for item in relevant_memories
        ])
    
    # Create messages array for OpenAI
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to stored memories. "
                                      "Use the provided memories when they are relevant to the user's question."}
    ]
    
    # Add memory context if available
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    
    # Add chat history
    messages.extend(chat_history)
    
    # Add current user message
    messages.append({"role": "user", "content": user_input})
    
    # Get LLM response
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Simple chat loop
async def chat_loop():
    chat_history = []
    print("Memory-augmented assistant (type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
            
        response = await chat_with_memory(user_input, chat_history)
        
        # Update chat history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        
        # Keep history to a reasonable size
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
            
        print(f"\nAssistant: {response}")

# Run the chat loop
if __name__ == "__main__":
    asyncio.run(chat_loop())
```

## Integration with Google Gemini

```python
import asyncio
import os
from memuri import Memuri
import google.generativeai as genai

# Initialize Memuri
memory = Memuri(
    vector_store="pgvector",
    vector_store_connection="postgresql://postgres:postgres@localhost:5432/memuri",
    embedder="google",
    embedder_model="models/embedding-001"
)
await memory.initialize()

# Initialize Google Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

async def chat_with_memory(user_input: str, chat_history):
    """Process a user message with memory-augmented responses using Gemini."""
    
    # Check if we should remember this message
    should_remember = any(keyword in user_input.lower() for keyword in 
                         ["remember", "note", "don't forget", "keep in mind"])
    
    if should_remember:
        # Store this in memory
        await memory.add_memory(content=user_input)
        print("[Saved to memory]")
    
    # Retrieve relevant memories
    relevant_memories = await memory.search_memory(user_input, limit=3)
    
    # Format memories as context for the LLM
    memory_context = ""
    if relevant_memories:
        memory_context = "Here are some relevant memories I have:\n" + "\n".join([
            f"- {item[0].content}" for item in relevant_memories
        ])
    
    # Prepare prompt with memory context
    prompt = ""
    if memory_context:
        prompt += memory_context + "\n\n"
    
    prompt += user_input
    
    # Get response from Gemini
    response = await asyncio.to_thread(
        model.generate_content,
        prompt,
        generation_config={"temperature": 0.7}
    )
    
    return response.text

# Chat loop similar to the OpenAI example
# ...
```

## Advanced Integration with Streaming and Memory Categories

```python
import asyncio
import os
import uuid
from memuri import Memuri
from openai import AsyncOpenAI
from datetime import datetime

# Initialize Memuri with custom categories
memory = Memuri(
    vector_store="pgvector",
    vector_store_connection="postgresql://postgres:postgres@localhost:5432/memuri",
    embedder="openai",
    memory_rules={
        "PERSONAL": {"threshold": 0.7, "action": "add"},
        "TASK": {"threshold": 0.6, "action": "add"},
        "QUESTION": {"action": "none"},  # Don't remember questions
        "FACT": {"action": "add", "ttl": 31536000}  # Facts last a year
    }
)
await memory.initialize()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class MemoryAugmentedChatbot:
    def __init__(self, memory, llm_client):
        self.memory = memory
        self.llm = llm_client
        self.conversation_id = str(uuid.uuid4())
        self.chat_history = []
        
    async def process_message(self, user_input: str, stream=True):
        """Process user message with memory augmentation."""
        
        # Determine memory category
        if user_input.startswith("Remember") or "note that" in user_input:
            # Detect memory category from content
            if any(kw in user_input.lower() for kw in ["meet", "appointment", "schedule"]):
                category = "TASK"
            elif any(kw in user_input.lower() for kw in ["likes", "favorite", "prefers"]):
                category = "PERSONAL"
            elif any(kw in user_input.lower() for kw in ["fact", "information", "data"]):
                category = "FACT"
            else:
                category = None  # Let auto-classifier decide
                
            # Store in memory with metadata
            await self.memory.add_memory(
                content=user_input,
                category=category,
                metadata={
                    "conversation_id": self.conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "source": "user_explicit"
                }
            )
            
        # Search both by raw query and classified intent
        raw_results = await self.memory.search_memory(user_input, limit=5)
        
        # Get intent-based memories if the query is about tasks
        intent_results = []
        if any(kw in user_input.lower() for kw in ["todo", "task", "schedule", "plan"]):
            intent_results = await self.memory.search_memory(
                user_input, 
                category="TASK",
                limit=3
            )
            
        # Combine and deduplicate results
        seen_ids = set()
        combined_results = []
        
        for doc, score in raw_results:
            if doc.id not in seen_ids:
                combined_results.append((doc, score))
                seen_ids.add(doc.id)
                
        for doc, score in intent_results:
            if doc.id not in seen_ids:
                combined_results.append((doc, score))
                seen_ids.add(doc.id)
                
        # Sort by score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        combined_results = combined_results[:5]  # Limit to top 5
        
        # Format memories as context
        memory_context = ""
        if combined_results:
            memory_context = "Relevant information from memory:\n"
            for i, (doc, score) in enumerate(combined_results, 1):
                memory_context += f"{i}. {doc.content}\n"
        
        # Create messages array for OpenAI
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to stored memories. "
                                         "Use the provided memories when relevant to the user's query."}
        ]
        
        # Add memory context if available
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        
        # Add chat history (last 5 exchanges)
        messages.extend(self.chat_history[-10:])
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": user_input})
        
        if stream:
            # Stream the response
            response_content = ""
            
            async def content_generator():
                nonlocal response_content
                
                response_stream = await self.llm.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                    stream=True
                )
                
                async for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response_content += content
                        yield content
            
            # Add placeholder for assistant's response
            self.chat_history.append({"role": "assistant", "content": ""})
            
            return content_generator(), lambda: self._finalize_response(response_content)
        else:
            # Non-streaming response
            response = await self.llm.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            
            response_content = response.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": response_content})
            
            return response_content
    
    async def _finalize_response(self, complete_response):
        """Finalize the streamed response in chat history."""
        if self.chat_history and self.chat_history[-1]["role"] == "assistant":
            self.chat_history[-1]["content"] = complete_response
            
        # Periodically store important exchanges in memory
        if (len(self.chat_history) >= 2 and 
            "important" in self.chat_history[-2]["content"].lower() or
            "remember" in self.chat_history[-2]["content"].lower()):
            
            exchange = f"User: {self.chat_history[-2]['content']}\nAssistant: {complete_response}"
            
            await self.memory.add_memory(
                content=exchange,
                metadata={
                    "conversation_id": self.conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "source": "important_exchange"
                }
            )

# Example usage in a FastAPI application
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

app = FastAPI()
chatbots = {}  # Store chatbot instances by session ID

@app.websocket("/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    # Get or create chatbot for this session
    if session_id not in chatbots:
        chatbots[session_id] = MemoryAugmentedChatbot(memory, client)
    
    chatbot = chatbots[session_id]
    
    try:
        while True:
            user_message = await websocket.receive_text()
            
            # Process with streaming
            stream_generator, finalize = await chatbot.process_message(user_message)
            
            # Stream response chunks to websocket
            async for chunk in stream_generator:
                await websocket.send_text(chunk)
            
            # Finalize response in chat history
            await finalize()
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup if needed
        pass

@app.post("/chat/{session_id}")
async def chat_endpoint(session_id: str, message: str):
    # Get or create chatbot for this session
    if session_id not in chatbots:
        chatbots[session_id] = MemoryAugmentedChatbot(memory, client)
    
    chatbot = chatbots[session_id]
    
    # Process with streaming for HTTP response
    stream_generator, finalize = await chatbot.process_message(message)
    
    async def response_stream():
        async for chunk in stream_generator:
            yield chunk
        
        # Finalize after streaming completes
        await finalize()
    
    return StreamingResponse(response_stream())
```

## Memory-Augmented RAG System

```python
import asyncio
import os
from memuri import Memuri
from openai import AsyncOpenAI
from typing import List, Dict, Any, Tuple
import json

class MemoryAugmentedRAG:
    """RAG system with memory capabilities to improve over time."""
    
    def __init__(self):
        self.memory = None
        self.llm_client = None
    
    async def initialize(self):
        """Initialize components."""
        # Initialize memory system
        self.memory = Memuri(
            vector_store="pgvector",
            vector_store_connection="postgresql://postgres:postgres@localhost:5432/memuri",
            embedder="openai",
            embedder_model="text-embedding-ada-002"
        )
        await self.memory.initialize()
        
        # Initialize LLM client
        self.llm_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    async def add_document(self, document: str, metadata: Dict[str, Any] = None):
        """Add a document to the system's knowledge base."""
        # Split into chunks (simplified - a real system would use better chunking)
        chunks = [document[i:i+1000] for i in range(0, len(document), 1000)]
        
        # Add each chunk to memory
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = i
            chunk_metadata["source"] = "document"
            
            await self.memory.add_memory(
                content=chunk,
                category="REFERENCE",
                metadata=chunk_metadata
            )
    
    async def process_query(self, query: str) -> Tuple[str, List[str]]:
        """Process a user query with RAG and memory tracking."""
        # Search memory for relevant context
        results = await self.memory.search_memory(query, limit=5)
        
        # Get the content from results
        context_docs = [doc.content for doc, _ in results]
        
        # Build prompt with retrieved context
        context_str = "\n\n".join(context_docs)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "system", "content": f"Context information:\n{context_str}"},
            {"role": "user", "content": query}
        ]
        
        # Get answer from LLM
        response = await self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        # Store the interaction in memory for future improvements
        interaction = {
            "query": query,
            "answer": answer,
            "sources": [doc.metadata.get("title", "Unknown") for doc, _ in results if hasattr(doc, "metadata")]
        }
        
        await self.memory.add_memory(
            content=json.dumps(interaction),
            category="INTERACTION",
            metadata={
                "type": "qa_pair",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return answer, [doc.content for doc, _ in results]
    
    async def improve_from_feedback(self, query: str, answer: str, feedback: Dict[str, Any]):
        """Store feedback to improve future responses."""
        await self.memory.log_feedback(
            content=json.dumps({
                "query": query,
                "answer": answer,
                "feedback": feedback
            }),
            category="FEEDBACK"
        )

# Usage example
async def main():
    rag = MemoryAugmentedRAG()
    await rag.initialize()
    
    # Add a document to the knowledge base
    await rag.add_document(
        "The Python programming language was created by Guido van Rossum and first released in 1991. "
        "It emphasizes code readability with its notable use of significant whitespace. "
        "Python features a dynamic type system and automatic memory management.",
        metadata={"title": "Python History", "source": "documentation"}
    )
    
    # Ask a question
    answer, sources = await rag.process_query("When was Python first released?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
    
    # Provide feedback
    await rag.improve_from_feedback(
        "When was Python first released?",
        answer,
        {"rating": 5, "comments": "Very accurate and concise"}
    )

if __name__ == "__main__":
    asyncio.run(main())
```

This example demonstrates multiple ways to integrate Memuri with LLMs to create powerful memory-augmented applications. 