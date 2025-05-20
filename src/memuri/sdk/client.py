"""High-level client for the memuri SDK."""

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional, Union

from memuri.core.config import MemuriSettings, EmbeddingSettings, LLMSettings, get_settings
from memuri.core.logging import configure_logging, get_logger
from memuri.core.telemetry import initialize_telemetry
from memuri.domain.interfaces import (
    ClassifierService,
    EmbeddingService,
    FeedbackService,
    LLMService,
    MemoryService,
    RerankingService,
)
from memuri.domain.models import (
    ChatMessage,
    Memory,
    MemoryCategory,
    MemorySource,
    SearchQuery,
    SearchResult,
)
from memuri.factory import (
    ClassifierFactory,
    EmbedderFactory,
    FeedbackServiceFactory,
    LlmFactory,
    VectorStoreFactory,
)
from memuri.services.memory import MemoryOrchestrator
from memuri.services.rerank import RerankService

logger = get_logger(__name__)


class Memuri:
    """Main client for the memuri SDK.
    
    This class provides a high-level interface to the memuri SDK, including
    memory operations, embedding, LLM interaction, and more.
    
    Example:
        ```python
        from memuri import Memuri
        
        # Create a client
        memuri = Memuri()
        
        # Add a memory
        memory = await memuri.add_memory(
            content="Remember to buy milk",
            category="TODO"
        )
        
        # Search for memories
        results = await memuri.search_memory("buy groceries")
        for memory in results.memories:
            print(f"{memory.score:.2f}: {memory.memory.content}")
        ```
    """
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> "Memuri":
        """Create a Memuri instance from a configuration dictionary.
        
        This method allows for easy configuration similar to the mem0 library,
        with a config dictionary specifying embedder and other settings.
        
        Example:
            ```python
            import os
            from memuri import Memuri
            
            os.environ["OPENAI_API_KEY"] = "your_api_key"
            
            config = {
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-ada-002",
                    }
                }
            }
            
            m = Memuri.from_config(config)
            ```
        
        Args:
            config: Configuration dictionary with providers and their settings
            
        Returns:
            Memuri: A configured Memuri instance
        """
        # Initialize with default settings
        settings = get_settings()
        
        # Process embedder configuration
        if "embedder" in config:
            embedder_config = config["embedder"]
            provider = embedder_config.get("provider")
            provider_config = embedder_config.get("config", {})
            
            # Create embedding settings
            embedding_settings = EmbeddingSettings(
                provider=provider,
                model_name=provider_config.get("model", "text-embedding-ada-002"),
                api_key=provider_config.get("api_key"),
                base_url=provider_config.get("base_url"),
                embedding_dims=provider_config.get("embedding_dims"),
                http_client_proxies=provider_config.get("http_client_proxies"),
                azure_kwargs=provider_config.get("azure_kwargs"),
                model_kwargs=provider_config.get("model_kwargs"),
                credentials_json=provider_config.get("credentials_json"),
            )
            
            # Update settings
            settings.embedding = embedding_settings
        
        # Process LLM configuration if provided
        if "llm" in config:
            llm_config = config["llm"]
            provider = llm_config.get("provider")
            provider_config = llm_config.get("config", {})
            
            # Create LLM settings
            llm_settings = LLMSettings(
                provider=provider,
                model_name=provider_config.get("model", "gpt-3.5-turbo"),
                api_key=provider_config.get("api_key"),
                base_url=provider_config.get("base_url"),
                temperature=provider_config.get("temperature", 0.7),
                max_tokens=provider_config.get("max_tokens", 800),
            )
            
            # Update settings
            settings.llm = llm_settings
        
        # Create Memuri instance with our configured settings
        return Memuri(settings=settings)
    
    def __init__(
        self,
        settings: Optional[MemuriSettings] = None,
        embedding_service: Optional[EmbeddingService] = None,
        memory_service: Optional[MemoryService] = None,
        llm_service: Optional[LLMService] = None,
        reranking_service: Optional[RerankingService] = None,
        classifier_service: Optional[ClassifierService] = None,
        feedback_service: Optional[FeedbackService] = None,
    ):
        """Initialize the memuri SDK client.
        
        Args:
            settings: Optional settings, if not provided, will be loaded from env/config
            embedding_service: Optional embedding service, if not provided, will be created
            memory_service: Optional memory service, if not provided, will be created
            llm_service: Optional LLM service, if not provided, will be created
            reranking_service: Optional reranking service, if not provided, will be created
            classifier_service: Optional classifier service, if not provided, will be created
            feedback_service: Optional feedback service, if not provided, will be created
        """
        # Initialize settings and logging
        self.settings = settings or get_settings()
        configure_logging(self.settings.logging)
        
        # Initialize services
        self._embedding_service = embedding_service or EmbedderFactory.create()
        self._memory_service = memory_service or VectorStoreFactory.create()
        self._llm_service = llm_service or LlmFactory.create()
        self._reranking_service = reranking_service or RerankService()
        self._classifier_service = classifier_service or ClassifierFactory.create()
        self._feedback_service = feedback_service or FeedbackServiceFactory.create()
        
        # Create memory orchestrator
        self._memory_orchestrator = MemoryOrchestrator(
            memory_service=self._memory_service,
            embedding_service=self._embedding_service,
            reranking_service=self._reranking_service,
            classifier_service=self._classifier_service,
            feedback_service=self._feedback_service,
            settings=self.settings,
        )
        
        # Initialize telemetry
        initialize_telemetry(self.settings.telemetry)
        
        logger.info("Memuri SDK initialized")
    
    async def add_memory(
        self,
        content: str,
        category: Optional[Union[str, MemoryCategory]] = None,
        source: Optional[Union[str, MemorySource]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Add a memory to the memory store.
        
        Args:
            content: Content of the memory
            category: Optional category, will be auto-classified if not provided
            source: Optional source of the memory
            metadata: Optional metadata to associate with the memory
            
        Returns:
            Memory: The added memory
        """
        # Convert string category to enum if needed
        if isinstance(category, str):
            category = MemoryCategory(category)
        
        # Convert string source to enum if needed
        if isinstance(source, str):
            source = MemorySource(source)
        
        return await self._memory_orchestrator.add_memory(
            content=content,
            category=category,
            source=source or MemorySource.USER,
            metadata=metadata or {},
        )
    
    async def search_memory(
        self,
        query: str,
        category: Optional[Union[str, MemoryCategory]] = None,
        top_k: int = 5,
        rerank: bool = True,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Search for memories.
        
        Args:
            query: Search query
            category: Optional category filter
            top_k: Number of results to return
            rerank: Whether to rerank results
            metadata_filters: Optional metadata filters
            
        Returns:
            SearchResult: Search results
        """
        # Convert string category to enum if needed
        if isinstance(category, str):
            category = MemoryCategory(category)
        
        # Create search query
        search_query = SearchQuery(
            query=query,
            category=category,
            top_k=top_k,
            rerank=rerank,
            metadata_filters=metadata_filters or {},
        )
        
        return await self._memory_orchestrator.search_memory(search_query)
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID.
        
        Args:
            memory_id: ID of the memory to get
            
        Returns:
            Optional[Memory]: Memory if found, None otherwise
        """
        return await self._memory_service.get(memory_id)
    
    async def update_memory(self, memory: Memory) -> Memory:
        """Update a memory.
        
        Args:
            memory: Memory to update
            
        Returns:
            Memory: Updated memory
        """
        return await self._memory_service.update(memory)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            bool: True if the memory was deleted, False otherwise
        """
        return await self._memory_service.delete(memory_id)
    
    async def count_memories(
        self,
        category: Optional[Union[str, MemoryCategory]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Count memories matching the given filters.
        
        Args:
            category: Optional category filter
            metadata_filters: Optional metadata filters
            
        Returns:
            int: Number of matching memories
        """
        # Convert string category to enum if needed
        if isinstance(category, str):
            category = MemoryCategory(category)
        
        return await self._memory_service.count(
            category=category,
            metadata_filters=metadata_filters,
        )
    
    async def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate text using the LLM service.
        
        Args:
            prompt: Prompt to generate from
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Union[str, AsyncIterator[str]]: Generated text or async iterator of chunks
        """
        if stream:
            return self._llm_service.generate_stream(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            return await self._llm_service.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
    
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate a response in a chat conversation.
        
        Args:
            messages: Chat history
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Union[str, AsyncIterator[str]]: Generated response or async iterator of chunks
        """
        if stream:
            return self._llm_service.chat_stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            return await self._llm_service.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using the embedding service.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List[List[float]]: Embeddings for the texts
        """
        response = await self._embedding_service.embed_texts(texts)
        return response.embeddings
    
    async def classify_text(self, text: str) -> Dict[MemoryCategory, float]:
        """Classify text into memory categories.
        
        Args:
            text: Text to classify
            
        Returns:
            Dict[MemoryCategory, float]: Categories with confidence scores
        """
        return await self._classifier_service.classify(text)
    
    async def log_feedback(
        self,
        text: str,
        category: Union[str, MemoryCategory],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log feedback for the classifier.
        
        Args:
            text: Text that was classified
            category: Correct category
            metadata: Optional metadata
        """
        # Convert string category to enum if needed
        if isinstance(category, str):
            category = MemoryCategory(category)
        
        await self._feedback_service.log_feedback(
            text=text,
            category=category,
            metadata=metadata,
        )
    
    async def close(self) -> None:
        """Close the client and release resources."""
        # TODO: Implement cleanup logic for services
        logger.info("Closing Memuri SDK client")


class AsyncIterator:
    """Utility type for type hinting."""
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        raise StopAsyncIteration 