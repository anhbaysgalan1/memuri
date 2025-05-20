"""Embedding service implementations."""

from typing import List, Optional, Dict, Any
import os

from openai import AsyncOpenAI
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from memuri.core.config import EmbeddingSettings
from memuri.core.logging import get_logger
from memuri.core.telemetry import embedding_latency, track_latency
from memuri.domain.interfaces import EmbeddingService
from memuri.domain.models import Document, EmbeddingResponse

logger = get_logger(__name__)


class BaseEmbeddingService:
    """Base class for embedding services."""
    
    def __init__(self, settings: EmbeddingSettings):
        """Initialize the embedding service.
        
        Args:
            settings: Embedding settings
        """
        self.settings = settings
        logger.info(f"Initialized {self.__class__.__name__} with model {settings.model_name}")
    
    @track_latency(embedding_latency)
    async def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        """Embed a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingResponse: Embeddings for the texts
        """
        raise NotImplementedError("Subclasses must implement embed_texts")
    
    async def embed_documents(self, documents: List[Document]) -> EmbeddingResponse:
        """Embed a list of documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            EmbeddingResponse: Embeddings for the documents
        """
        # Extract text content from documents
        texts = [doc.content for doc in documents]
        return await self.embed_texts(texts)
    
    async def get_dimensions(self) -> int:
        """Get the dimensions of the embeddings produced by this service.
        
        Returns:
            int: Embedding dimensions
        """
        raise NotImplementedError("Subclasses must implement get_dimensions")


class OpenAIEmbeddingService(BaseEmbeddingService):
    """OpenAI embedding service."""
    
    def __init__(self, settings: EmbeddingSettings):
        """Initialize the OpenAI embedding service.
        
        Args:
            settings: Embedding settings
        """
        super().__init__(settings)
        
        # Configure client with settings
        client_kwargs: Dict[str, Any] = {}
        
        # Use API key from settings or fall back to environment
        if settings.api_key:
            client_kwargs["api_key"] = settings.api_key
            
        # Use base URL if provided
        if settings.base_url:
            client_kwargs["base_url"] = settings.base_url
            
        # Add Azure-specific configuration if provided
        if settings.azure_kwargs:
            client_kwargs.update(settings.azure_kwargs)
            
        # Initialize client with our settings
        self.client = AsyncOpenAI(**client_kwargs)
        
        # Model dimension mapping
        self.dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
    
    @track_latency(embedding_latency, {"provider": "openai"})
    async def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        """Embed texts using OpenAI's API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingResponse: Embeddings for the texts
        """
        # Process in batches to respect settings.batch_size
        batch_size = self.settings.batch_size
        all_embeddings = []
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = await self.client.embeddings.create(
                model=self.settings.model_name,
                input=batch,
            )
            
            # Extract embeddings from response
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            total_tokens += response.usage.total_tokens
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=self.settings.model_name,
            dimensions=await self.get_dimensions(),
            tokens=total_tokens,
        )
    
    async def get_dimensions(self) -> int:
        """Get the dimensions of the embeddings produced by this service.
        
        Returns:
            int: Embedding dimensions
        """
        # Use explicitly configured dimensions if provided
        if self.settings.embedding_dims:
            return self.settings.embedding_dims
            
        # Otherwise check our known models
        if self.settings.model_name in self.dimensions:
            return self.dimensions[self.settings.model_name]
            
        # Default to settings dimension as fallback
        return 1536


class GoogleEmbeddingService(BaseEmbeddingService):
    """Google embedding service."""
    
    def __init__(self, settings: EmbeddingSettings):
        """Initialize the Google embedding service.
        
        Args:
            settings: Embedding settings
        """
        super().__init__(settings)
        
        # Configure Google client with API key from settings
        if settings.api_key:
            genai.configure(api_key=settings.api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            # Fallback to environment variable
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        
        # Model dimension mapping
        self.dimensions = {
            "models/embedding-001": 768,
            "models/text-embedding-004": 768,
            "gemini-embedding-exp-03-07": 1536,
        }
        
        self.model = settings.model_name
        
        # Set task type based on model
        self.task_type = "retrieval_document"
        if settings.model_kwargs and "task_type" in settings.model_kwargs:
            self.task_type = settings.model_kwargs["task_type"]
        
        logger.info(f"Initialized Google embedding service with model {self.model}")
    
    @track_latency(embedding_latency, {"provider": "google"})
    async def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        """Embed texts using Google's API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingResponse: Embeddings for the texts
        """
        batch_size = self.settings.batch_size
        all_embeddings = []
        total_tokens = 0  # Google doesn't provide token count directly
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process each text individually and collect embeddings
            for text in batch:
                try:
                    # Use async wrapper to avoid blocking (Google SDK is sync)
                    embedding_result = await self._get_embedding(text)
                    all_embeddings.append(embedding_result['values'])
                    total_tokens += len(text.split())  # Rough estimation of tokens
                except Exception as e:
                    logger.error(f"Error embedding text with Google API: {e}")
                    # Return zero vector as fallback
                    all_embeddings.append([0.0] * await self.get_dimensions())
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=self.settings.model_name,
            dimensions=await self.get_dimensions(),
            tokens=total_tokens,
        )
    
    async def _get_embedding(self, text: str) -> Dict[str, Any]:
        """Get embedding for a single text using Google API.
        
        This method is implemented to wrap the synchronous Google API
        in an async context.
        
        Args:
            text: Text to embed
            
        Returns:
            Dict: Embedding response from Google
        """
        import asyncio
        
        try:
            # Run embed_content in an executor to avoid blocking the event loop
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type=self.task_type,
                )
            )
            
            # Ensure we have the expected structure
            if "embedding" not in result:
                raise ValueError(f"Unexpected response format from Google API: {result}")
                
            return result["embedding"]
        except Exception as e:
            logger.error(f"Error calling Google embedding API: {e}")
            raise
    
    async def get_dimensions(self) -> int:
        """Get the dimensions of the embeddings produced by this service.
        
        Returns:
            int: Embedding dimensions
        """
        # Use explicitly configured dimensions if provided
        if self.settings.embedding_dims:
            return self.settings.embedding_dims
            
        # Otherwise check our known models
        if self.settings.model_name in self.dimensions:
            return self.dimensions[self.settings.model_name]
            
        # Default to 768 for Google models as fallback
        return 768


class SentenceTransformersEmbeddingService(BaseEmbeddingService):
    """SentenceTransformers embedding service."""
    
    def __init__(self, settings: EmbeddingSettings):
        """Initialize the SentenceTransformers embedding service.
        
        Args:
            settings: Embedding settings
        """
        super().__init__(settings)
        # Initialize the model
        self.model = None  # Lazy-loaded on first use
        self.model_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "all-distilroberta-v1": 768,
            "multi-qa-MiniLM-L6-dot-v1": 384,
        }
        
        # Device settings
        self.device = settings.model_kwargs.get("device", "cpu") if settings.model_kwargs else "cpu"
    
    def _ensure_model_loaded(self):
        """Ensure the model is loaded."""
        if self.model is None:
            self.model = SentenceTransformer(self.settings.model_name, device=self.device)
    
    @track_latency(embedding_latency, {"provider": "sentence_transformers"})
    async def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        """Embed texts using SentenceTransformers.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingResponse: Embeddings for the texts
        """
        import asyncio
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Run embedding in a thread to avoid blocking the event loop
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.model.encode(texts, batch_size=self.settings.batch_size)
        )
        
        # Convert to list of lists
        embeddings_list = embeddings.tolist()
        
        # Estimate token count (very rough estimate)
        total_tokens = sum(len(text.split()) for text in texts)
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            model=self.settings.model_name,
            dimensions=await self.get_dimensions(),
            tokens=total_tokens,
        )
    
    async def get_dimensions(self) -> int:
        """Get the dimensions of the embeddings produced by this service.
        
        Returns:
            int: Embedding dimensions
        """
        # Use explicitly configured dimensions if provided
        if self.settings.embedding_dims:
            return self.settings.embedding_dims
            
        # Check our known models
        if self.settings.model_name in self.model_dimensions:
            return self.model_dimensions[self.settings.model_name]
            
        # If model is loaded, get dimensions directly
        if self.model is not None:
            return self.model.get_sentence_embedding_dimension()
            
        # Load model to get dimensions
        self._ensure_model_loaded()
        return self.model.get_sentence_embedding_dimension() 