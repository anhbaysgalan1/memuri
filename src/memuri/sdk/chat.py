"""Chat utilities for the memuri SDK."""

from typing import Any, Dict, List, Optional, Union

from memuri.domain.models import ChatMessage, Memory, MessageRole, SearchResult
from memuri.sdk.client import Memuri


class Chat:
    """Helper class for managing chat conversations with memory integration.
    
    This class provides a convenience wrapper around the Memuri client
    for chat-based use cases, including automatic memory retrieval and
    message history management.
    
    Example:
        ```python
        from memuri.sdk.chat import Chat
        
        # Create a chat manager
        chat = Chat()
        
        # Add messages
        await chat.add_user_message("What's the capital of France?")
        response = await chat.generate_response()
        
        # Response will include relevant memories
        print(response)
        
        # Add the response to history
        await chat.add_assistant_message(response)
        ```
    """
    
    def __init__(
        self,
        memuri: Optional[Memuri] = None,
        system_prompt: Optional[str] = None,
        auto_memory_retrieval: bool = True,
        memory_retrieval_count: int = 3,
    ):
        """Initialize the chat manager.
        
        Args:
            memuri: Optional Memuri client, if not provided, will be created
            system_prompt: Optional system prompt to include in the chat
            auto_memory_retrieval: Whether to automatically retrieve memories
            memory_retrieval_count: Number of memories to retrieve
        """
        self.memuri = memuri or Memuri()
        self.messages: List[ChatMessage] = []
        self.auto_memory_retrieval = auto_memory_retrieval
        self.memory_retrieval_count = memory_retrieval_count
        
        # Add system prompt if provided
        if system_prompt:
            self.add_system_message(system_prompt)
    
    def add_system_message(self, content: str) -> ChatMessage:
        """Add a system message to the chat.
        
        Args:
            content: Message content
            
        Returns:
            ChatMessage: The added message
        """
        message = ChatMessage(role=MessageRole.SYSTEM, content=content)
        self.messages.append(message)
        return message
    
    async def add_user_message(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        store_as_memory: bool = False,
    ) -> ChatMessage:
        """Add a user message to the chat.
        
        Args:
            content: Message content
            metadata: Optional metadata to associate with the message
            store_as_memory: Whether to store the message as a memory
            
        Returns:
            ChatMessage: The added message
        """
        message = ChatMessage(
            role=MessageRole.USER, 
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(message)
        
        # Store as memory if requested
        if store_as_memory:
            await self.memuri.add_memory(
                content=content,
                metadata={"from_chat": True, **(metadata or {})},
            )
        
        return message
    
    async def add_assistant_message(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Add an assistant message to the chat.
        
        Args:
            content: Message content
            metadata: Optional metadata to associate with the message
            
        Returns:
            ChatMessage: The added message
        """
        message = ChatMessage(
            role=MessageRole.ASSISTANT, 
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(message)
        return message
    
    async def generate_response(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        retrieve_memories: Optional[bool] = None,
    ) -> Union[str, AsyncIterator]:
        """Generate a response to the current chat.
        
        Args:
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            retrieve_memories: Whether to retrieve memories before generating
                (overrides auto_memory_retrieval)
            
        Returns:
            Union[str, AsyncIterator]: The generated response or an async iterator of chunks
        """
        # Determine whether to retrieve memories
        should_retrieve = retrieve_memories if retrieve_memories is not None else self.auto_memory_retrieval
        
        # Check if we have at least one user message
        if not any(msg.role == MessageRole.USER for msg in self.messages):
            raise ValueError("Cannot generate a response without a user message")
        
        # Retrieve relevant memories if requested
        memories: Optional[List[Memory]] = None
        if should_retrieve:
            # Get the last user message as the query
            query = next(
                msg.content for msg in reversed(self.messages) 
                if msg.role == MessageRole.USER
            )
            
            # Search for memories
            search_result = await self.memuri.search_memory(
                query=query,
                top_k=self.memory_retrieval_count,
            )
            
            # Extract memories
            memories = [m.memory for m in search_result.memories]
        
        # Prepare messages with memory context if found
        context_messages = list(self.messages)
        
        # Add memory context if retrieved
        if memories and len(memories) > 0:
            # Create a system message with memory context
            memory_context = "\n\n".join([
                f"Memory {i+1}: {memory.content}" 
                for i, memory in enumerate(memories)
            ])
            
            memory_message = ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"Relevant memories:\n{memory_context}",
                metadata={"from_memory": True},
            )
            
            # Insert at the position just before the last user message
            for i in range(len(context_messages) - 1, -1, -1):
                if context_messages[i].role == MessageRole.USER:
                    context_messages.insert(i, memory_message)
                    break
        
        # Generate response
        return await self.memuri.chat(
            messages=context_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
    
    async def clear_history(self) -> None:
        """Clear the chat history.
        
        This will remove all messages from the chat history.
        """
        self.messages = []
    
    async def close(self) -> None:
        """Close the chat manager and release resources."""
        await self.memuri.close()


class AsyncIterator:
    """Utility type for type hinting."""
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        raise StopAsyncIteration 