"""LLM service implementations for text generation."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from openai import AsyncOpenAI
import google.generativeai as genai

from memuri.core.config import LLMSettings
from memuri.core.logging import get_logger
from memuri.core.telemetry import llm_request_duration, track_latency
from memuri.domain.interfaces import LLMService
from memuri.domain.models import ChatMessage, MessageRole

logger = get_logger(__name__)


class BaseLLMService:
    """Base class for LLM services."""
    
    def __init__(self, settings: LLMSettings):
        """Initialize the LLM service.
        
        Args:
            settings: LLM settings
        """
        self.settings = settings
        logger.info(f"Initialized {self.__class__.__name__} with model {settings.model_name}")
    
    @track_latency(llm_request_duration)
    async def generate(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Prompt to generate from
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Returns:
            str: Generated text
        """
        raise NotImplementedError("Subclasses must implement generate")
    
    async def generate_stream(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text from a prompt as a stream.
        
        Args:
            prompt: Prompt to generate from
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Yields:
            str: Generated text chunks
        """
        raise NotImplementedError("Subclasses must implement generate_stream")
    
    @track_latency(llm_request_duration)
    async def chat(
        self, 
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate a response in a chat conversation.
        
        Args:
            messages: Chat history
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Returns:
            str: Generated response
        """
        raise NotImplementedError("Subclasses must implement chat")
    
    async def chat_stream(
        self, 
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate a response in a chat conversation as a stream.
        
        Args:
            messages: Chat history
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Yields:
            str: Generated response chunks
        """
        raise NotImplementedError("Subclasses must implement chat_stream")


class OpenAILLMService(BaseLLMService):
    """OpenAI LLM service."""
    
    def __init__(self, settings: LLMSettings):
        """Initialize the OpenAI LLM service.
        
        Args:
            settings: LLM settings
        """
        super().__init__(settings)
        self.client = AsyncOpenAI()  # Assumes API key is in environment
    
    @track_latency(llm_request_duration, {"provider": "openai"})
    async def generate(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text from a prompt using OpenAI.
        
        Args:
            prompt: Prompt to generate from
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Returns:
            str: Generated text
        """
        # Convert prompt to a single system message
        messages = [{"role": "system", "content": prompt}]
        
        # Use settings if parameters not provided
        temp = temperature if temperature is not None else self.settings.temperature
        tokens = max_tokens if max_tokens is not None else self.settings.max_tokens
        
        response = await self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            stop=stop_sequences,
        )
        
        return response.choices[0].message.content or ""
    
    async def generate_stream(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text from a prompt as a stream using OpenAI.
        
        Args:
            prompt: Prompt to generate from
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Yields:
            str: Generated text chunks
        """
        # Convert prompt to a single system message
        messages = [{"role": "system", "content": prompt}]
        
        # Use settings if parameters not provided
        temp = temperature if temperature is not None else self.settings.temperature
        tokens = max_tokens if max_tokens is not None else self.settings.max_tokens
        
        response = await self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            stop=stop_sequences,
            stream=True,
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    @track_latency(llm_request_duration, {"provider": "openai"})
    async def chat(
        self, 
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate a response in a chat conversation using OpenAI.
        
        Args:
            messages: Chat history
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Returns:
            str: Generated response
        """
        # Convert our ChatMessage objects to OpenAI format
        openai_messages = [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in messages
        ]
        
        # Use settings if parameters not provided
        temp = temperature if temperature is not None else self.settings.temperature
        tokens = max_tokens if max_tokens is not None else self.settings.max_tokens
        
        response = await self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=openai_messages,
            temperature=temp,
            max_tokens=tokens,
            stop=stop_sequences,
        )
        
        return response.choices[0].message.content or ""
    
    async def chat_stream(
        self, 
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate a response in a chat conversation as a stream using OpenAI.
        
        Args:
            messages: Chat history
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Yields:
            str: Generated response chunks
        """
        # Convert our ChatMessage objects to OpenAI format
        openai_messages = [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in messages
        ]
        
        # Use settings if parameters not provided
        temp = temperature if temperature is not None else self.settings.temperature
        tokens = max_tokens if max_tokens is not None else self.settings.max_tokens
        
        response = await self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=openai_messages,
            temperature=temp,
            max_tokens=tokens,
            stop=stop_sequences,
            stream=True,
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class GoogleLLMService(BaseLLMService):
    """Google LLM service."""
    
    def __init__(self, settings: LLMSettings):
        """Initialize the Google LLM service.
        
        Args:
            settings: LLM settings
        """
        super().__init__(settings)
        # TODO: Initialize Google client
    
    @track_latency(llm_request_duration, {"provider": "google"})
    async def generate(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text from a prompt using Google.
        
        Args:
            prompt: Prompt to generate from
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Returns:
            str: Generated text
        """
        # TODO: Implement Google LLM API call
        # This is a placeholder implementation
        return f"[Generated text from Google for prompt: {prompt[:10]}...]"
    
    async def generate_stream(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text from a prompt as a stream using Google.
        
        Args:
            prompt: Prompt to generate from
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Yields:
            str: Generated text chunks
        """
        # TODO: Implement Google LLM streaming API call
        # This is a placeholder implementation
        yield f"[Generated text from Google for prompt: {prompt[:10]}...]"
    
    @track_latency(llm_request_duration, {"provider": "google"})
    async def chat(
        self, 
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate a response in a chat conversation using Google.
        
        Args:
            messages: Chat history
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Returns:
            str: Generated response
        """
        # TODO: Implement Google LLM API call
        # This is a placeholder implementation
        last_message = next((m for m in reversed(messages) if m.role == MessageRole.USER), None)
        return f"[Generated response from Google for message: {last_message.content[:10] if last_message else 'None'}...]"
    
    async def chat_stream(
        self, 
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate a response in a chat conversation as a stream using Google.
        
        Args:
            messages: Chat history
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Yields:
            str: Generated response chunks
        """
        # TODO: Implement Google LLM streaming API call
        # This is a placeholder implementation
        last_message = next((m for m in reversed(messages) if m.role == MessageRole.USER), None)
        yield f"[Generated response from Google for message: {last_message.content[:10] if last_message else 'None'}...]" 