"""Factory classes for dynamically loading providers."""

import importlib
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast, Protocol, runtime_checkable

from memuri.core.config import EmbeddingSettings, LLMSettings, VectorStoreSettings, get_settings
from memuri.core.logging import get_logger
from memuri.domain.interfaces import EmbeddingService, LLMService, MemoryService

logger = get_logger(__name__)

# Type variable for provider classes
T = TypeVar("T")


@runtime_checkable
class ClassifierServiceProtocol(Protocol):
    """Protocol for classifier services."""
    
    async def classify(self, text: str) -> Dict[Any, float]:
        """Classify text.
        
        Args:
            text: Text to classify
            
        Returns:
            Dict[Any, float]: Classification results with scores
        """
        ...


@runtime_checkable
class FeedbackServiceProtocol(Protocol):
    """Protocol for feedback services."""
    
    async def log_feedback(self, text: str, category: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log feedback.
        
        Args:
            text: Text to classify
            category: Correct category
            metadata: Optional metadata
        """
        ...


class ProviderFactory:
    """Base factory class for provider selection and instantiation."""
    
    # Maps provider names to implementation class paths
    provider_map: Dict[str, str] = {}
    
    # The base class that all providers must implement
    base_class: Type[Any] = object
    
    @classmethod
    def get_provider_class(cls, provider_name: str) -> Type[T]:
        """Get the provider class for the given provider name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Type: Provider class
            
        Raises:
            ValueError: If the provider is not supported
        """
        if provider_name not in cls.provider_map:
            supported = ", ".join(cls.provider_map.keys())
            raise ValueError(f"Unsupported provider: {provider_name}. Supported: {supported}")
        
        # Get the module and class name
        module_path = cls.provider_map[provider_name]
        module_name, class_name = module_path.rsplit(".", 1)
        
        try:
            # Import the module and get the class
            module = importlib.import_module(module_name)
            provider_class = getattr(module, class_name)
            
            # Verify that the class implements the correct interface
            if not issubclass(provider_class, cls.base_class):
                raise ValueError(
                    f"Provider {provider_name} does not implement {cls.base_class.__name__}"
                )
            
            return cast(Type[T], provider_class)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load provider {provider_name}: {e}")
            raise ValueError(f"Failed to load provider {provider_name}: {e}") from e


class EmbedderFactory(ProviderFactory):
    """Factory for creating embedding service instances."""
    
    provider_map = {
        "openai": "memuri.services.embedding.OpenAIEmbeddingService",
        "google": "memuri.services.embedding.GoogleEmbeddingService",
        "sentence_transformers": "memuri.services.embedding.SentenceTransformersEmbeddingService",
    }
    
    base_class = EmbeddingService
    
    @classmethod
    def create(
        cls, 
        provider: Optional[str] = None, 
        settings: Optional[EmbeddingSettings] = None
    ) -> EmbeddingService:
        """Create an embedding service instance.
        
        Args:
            provider: Name of the provider, defaults to config
            settings: Embedding settings, defaults to global config
            
        Returns:
            EmbeddingService: An initialized embedding service
        """
        # Use settings from config if not provided
        if settings is None:
            settings = get_settings().embedding
        
        # Use provider from settings if not provided
        if provider is None:
            provider = settings.provider
        
        # Get the provider class
        provider_class = cls.get_provider_class(provider)
        
        # Create and return an instance
        return provider_class(settings=settings)


class LlmFactory(ProviderFactory):
    """Factory for creating LLM service instances."""
    
    provider_map = {
        "openai": "memuri.services.llm.OpenAILLMService",
        "google": "memuri.services.llm.GoogleLLMService",
    }
    
    base_class = LLMService
    
    @classmethod
    def create(
        cls, 
        provider: Optional[str] = None, 
        settings: Optional[LLMSettings] = None
    ) -> LLMService:
        """Create an LLM service instance.
        
        Args:
            provider: Name of the provider, defaults to config
            settings: LLM settings, defaults to global config
            
        Returns:
            LLMService: An initialized LLM service
        """
        # Use settings from config if not provided
        if settings is None:
            settings = get_settings().llm
        
        # Use provider from settings if not provided
        if provider is None:
            provider = settings.provider
        
        # Get the provider class
        provider_class = cls.get_provider_class(provider)
        
        # Create and return an instance
        return provider_class(settings=settings)


class VectorStoreFactory(ProviderFactory):
    """Factory for creating vector store instances."""
    
    provider_map = {
        "pgvector": "memuri.adapters.vectorstore.pgvector.PgVectorStore",
        "milvus": "memuri.adapters.vectorstore.milvus.MilvusVectorStore",
        "qdrant": "memuri.adapters.vectorstore.qdrant.QdrantVectorStore",
        "redis_vector": "memuri.adapters.vectorstore.redis_vector.RedisVectorStore",
    }
    
    base_class = MemoryService
    
    @classmethod
    def create(
        cls, 
        provider: Optional[str] = None, 
        settings: Optional[VectorStoreSettings] = None
    ) -> MemoryService:
        """Create a vector store instance.
        
        Args:
            provider: Name of the provider, defaults to config
            settings: Vector store settings, defaults to global config
            
        Returns:
            MemoryService: An initialized vector store
        """
        # Use settings from config if not provided
        if settings is None:
            settings = get_settings().vector_store
        
        # Use provider from settings if not provided
        if provider is None:
            provider = settings.provider
        
        # Get the provider class
        provider_class = cls.get_provider_class(provider)
        
        # Create and return an instance
        return provider_class(settings=settings)


class ClassifierFactory(ProviderFactory):
    """Factory for creating classifier service instances."""
    
    provider_map = {
        "keyword": "memuri.services.classifier.KeywordClassifier",
        "ml": "memuri.services.classifier.MLClassifier",
    }
    
    base_class = ClassifierServiceProtocol
    
    @classmethod
    def create(cls, provider: str = "keyword") -> ClassifierServiceProtocol:
        """Create a classifier service instance.
        
        Args:
            provider: Name of the provider
            
        Returns:
            ClassifierServiceProtocol: An initialized classifier service
        """
        provider_class = cls.get_provider_class(provider)
        return provider_class()


class FeedbackServiceFactory:
    """Factory for creating feedback service instances."""
    
    @classmethod
    def create(cls) -> FeedbackServiceProtocol:
        """Create a feedback service instance.
        
        Returns:
            FeedbackServiceProtocol: An initialized feedback service
        """
        from memuri.services.feedback import FeedbackService
        return FeedbackService(settings=get_settings().feedback) 