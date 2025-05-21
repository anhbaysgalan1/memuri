"""Embedding services for the memuri SDK."""

import os
from typing import List, Optional, Dict, Any

import numpy as np
from openai import AsyncOpenAI

from memuri.core.config import EmbeddingSettings
from memuri.core.logging import get_logger
from memuri.core.telemetry import track_latency
from memuri.domain.models import Document, EmbeddingResponse
from memuri.core.text_utils import normalize_text_for_embedding, batch_text

logger = get_logger(__name__)


class OpenAIEmbeddingService:
    """OpenAI embedding service."""
    
    def __init__(self, settings: Optional[EmbeddingSettings] = None):
        """Initialize the OpenAI embedding service.
        
        Args:
            settings: Optional embedding settings
        """
        # Use provided settings or create default
        self.settings = settings or EmbeddingSettings()
        
        # Get API key from settings or environment
        api_key = self.settings.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        # Initialize the client with all relevant settings
        self._init_client(api_key)
        
        # Set model name
        self.model = self.settings.model_name
        
        logger.info(f"Initialized OpenAIEmbeddingService with model {self.model}")
    
    def _init_client(self, api_key: str) -> None:
        """Initialize the OpenAI client with the appropriate settings.
        
        Args:
            api_key: The OpenAI API key
        """
        # Start with basic required parameters
        client_params: Dict[str, Any] = {
            "api_key": api_key,
        }
        
        # Add base_url if provided
        if self.settings.base_url:
            client_params["base_url"] = self.settings.base_url
        
        # Add Azure kwargs if provided
        if self.settings.azure_kwargs:
            client_params.update(self.settings.azure_kwargs)
        
        # Add HTTP proxies if provided
        if self.settings.http_client_proxies:
            client_params["http_client"] = {
                "proxies": self.settings.http_client_proxies
            }
            
        # Add other model kwargs if provided
        if self.settings.model_kwargs:
            for key, value in self.settings.model_kwargs.items():
                # Don't override existing params
                if key not in client_params:
                    client_params[key] = value
        
        try:
            # Initialize the client with all parameters
            self.client = AsyncOpenAI(**client_params)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ValueError(f"Failed to initialize OpenAI client: {e}") from e
    
    def _sanitize_text_for_embedding(self, texts: List[str]) -> List[str]:
        """Sanitize text for embedding to prevent API errors.
        
        Args:
            texts: List of texts to sanitize
            
        Returns:
            List of sanitized texts
        """
        sanitized_texts = []
        for text in texts:
            # Skip empty texts
            if not text or text.strip() == "":
                sanitized_texts.append("")
                continue
                
            # Normalize and sanitize text
            try:
                sanitized = normalize_text_for_embedding(text)
                # Ensure we have at least some content
                if not sanitized or sanitized.strip() == "":
                    logger.warning(f"Text was sanitized to empty string: '{text[:50]}...'")
                    sanitized = "Empty content after sanitization"
                sanitized_texts.append(sanitized)
            except Exception as e:
                logger.error(f"Error sanitizing text: {e}")
                # Add a placeholder to maintain index alignment
                sanitized_texts.append("Error in text content")
        
        return sanitized_texts
    
    @track_latency()
    async def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        """Embed a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingResponse: Embeddings for the texts
        """
        # Handle empty list
        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                model=self.model,
                dimensions=self.settings.embedding_dims or 1536,
                tokens=0,
            )
        
        # Sanitize texts for embedding to prevent API errors
        sanitized_texts = self._sanitize_text_for_embedding(texts)
            
        try:
            # Call OpenAI API with any extra parameters
            embedding_params = {
                "model": self.model,
                "input": sanitized_texts,
            }
            
            # Add any additional model parameters if provided
            if self.settings.model_kwargs:
                for key, value in self.settings.model_kwargs.items():
                    if key not in embedding_params:
                        embedding_params[key] = value
            
            response = await self.client.embeddings.create(**embedding_params)
            
            # Extract embeddings
            data = response.data
            embeddings = [item.embedding for item in data]
            
            # Calculate total tokens
            usage = response.usage
            total_tokens = usage.total_tokens if usage else 0
            
            # Get dimensions
            dimensions = len(embeddings[0]) if embeddings else self.settings.embedding_dims or 1536
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=self.model,
                dimensions=dimensions,
                tokens=total_tokens,
            )
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # For debugging
            if len(sanitized_texts) < 5:
                logger.debug(f"Attempted to embed texts: {sanitized_texts}")
            else:
                logger.debug(f"Attempted to embed {len(sanitized_texts)} texts")
            raise ValueError(f"Error generating embeddings: {e}") from e
    
    async def embed_documents(self, documents: List[Document]) -> EmbeddingResponse:
        """Embed a list of documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            EmbeddingResponse: Embeddings for the documents
        """
        # Extract document content
        texts = [doc.content for doc in documents]
        
        # Embed texts
        return await self.embed_texts(texts)
    
    async def get_dimensions(self) -> int:
        """Get the dimensions of the embeddings produced by this service.
        
        Returns:
            int: Embedding dimensions
        """
        # Use configured dimensions or default
        return self.settings.embedding_dims or 1536 