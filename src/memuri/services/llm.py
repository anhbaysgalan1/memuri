"""LLM services for memuri."""

import os
from typing import List, Optional, Dict, Any

from openai import AsyncOpenAI

from memuri.core.config import LLMSettings
from memuri.core.logging import get_logger
from memuri.core.telemetry import track_latency
from memuri.domain.models import ChatMessage, MessageRole

logger = get_logger(__name__)


class OpenAILLMService:
    """OpenAI LLM service."""
    
    def __init__(self, settings: Optional[LLMSettings] = None):
        """Initialize the OpenAI LLM service.
        
        Args:
            settings: Optional LLM settings
        """
        # Use provided settings or create default
        self.settings = settings or LLMSettings()
        
        # Get API key from settings or environment
        api_key = self.settings.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        # Set up client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.settings.base_url,
        )
        
        # Set model name and default parameters
        self.model = self.settings.model_name
        self.temperature = self.settings.temperature
        self.max_tokens = self.settings.max_tokens
        
        logger.info(f"Initialized OpenAILLMService with model {self.model}")
    
    @track_latency()
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
        # Convert prompt to messages format
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Call OpenAI API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop_sequences,
        )
        
        # Extract generated text
        return response.choices[0].message.content
    
    async def generate_stream(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ):
        """Generate text from a prompt as a stream.
        
        Args:
            prompt: Prompt to generate from
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Yields:
            str: Generated text chunks
        """
        # Convert prompt to messages format
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Call OpenAI API with streaming
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop_sequences,
            stream=True,
        )
        
        # Yield chunks as they come in
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    @track_latency()
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
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        # Call OpenAI API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop_sequences,
        )
        
        # Extract generated text
        return response.choices[0].message.content
    
    async def chat_stream(
        self, 
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ):
        """Generate a response in a chat conversation as a stream.
        
        Args:
            messages: Chat history
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Yields:
            str: Generated response chunks
        """
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        # Call OpenAI API with streaming
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop_sequences,
            stream=True,
        )
        
        # Yield chunks as they come in
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
                
class GoogleLLMService:
    """Google Gemini LLM service stub."""
    
    def __init__(self, settings: Optional[LLMSettings] = None):
        """Initialize the Google LLM service.
        
        Args:
            settings: Optional LLM settings
        """
        # This is a stub implementation - would use Google Generative AI in production
        self.settings = settings or LLMSettings()
        logger.info(f"Initialized GoogleLLMService with model {self.settings.model_name}")
        
    async def generate(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text from a prompt.
        
        This is a stub implementation.
        
        Args:
            prompt: Prompt to generate from
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Returns:
            str: Generated text
        """
        return f"This is a stub response from Google Gemini: {prompt[:20]}..."
    
    async def generate_stream(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ):
        """Generate text from a prompt as a stream.
        
        This is a stub implementation.
        
        Args:
            prompt: Prompt to generate from
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Yields:
            str: Generated text chunks
        """
        words = f"This is a stub response from Google Gemini: {prompt[:20]}...".split()
        for word in words:
            yield word + " "
    
    async def chat(
        self, 
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate a response in a chat conversation.
        
        This is a stub implementation.
        
        Args:
            messages: Chat history
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Returns:
            str: Generated response
        """
        # Get the last user message, or use a placeholder
        last_message = "Hello"
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                last_message = msg.content
                break
                
        return f"This is a stub response from Google Gemini to: {last_message[:20]}..."
    
    async def chat_stream(
        self, 
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ):
        """Generate a response in a chat conversation as a stream.
        
        This is a stub implementation.
        
        Args:
            messages: Chat history
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Sequences that should stop generation
            
        Yields:
            str: Generated response chunks
        """
        # Get the last user message, or use a placeholder
        last_message = "Hello"
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                last_message = msg.content
                break
                
        words = f"This is a stub response from Google Gemini to: {last_message[:20]}...".split()
        for word in words:
            yield word + " " 