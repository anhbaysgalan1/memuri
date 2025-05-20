"""Interfaces for the memuri SDK."""

import abc
from typing import Any, Dict, List, Optional, Protocol, Union

from memuri.domain.models import (
    ChatMessage,
    Document,
    EmbeddingResponse,
    Memory,
    MemoryCategory,
    SearchQuery,
    SearchResult,
    ScoredMemory,
)


class EmbeddingService(Protocol):
    """Interface for embedding services."""
    
    @abc.abstractmethod
    async def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        """Embed a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingResponse: Embeddings for the texts
        """
        ...
    
    @abc.abstractmethod
    async def embed_documents(self, documents: List[Document]) -> EmbeddingResponse:
        """Embed a list of documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            EmbeddingResponse: Embeddings for the documents
        """
        ...
    
    @abc.abstractmethod
    async def get_dimensions(self) -> int:
        """Get the dimensions of the embeddings produced by this service.
        
        Returns:
            int: Embedding dimensions
        """
        ...


class MemoryService(Protocol):
    """Interface for memory storage and retrieval."""
    
    @abc.abstractmethod
    async def add(self, memory: Memory) -> Memory:
        """Add a memory to the store.
        
        Args:
            memory: Memory to add
            
        Returns:
            Memory: Added memory with ID and any other fields populated
        """
        ...
    
    @abc.abstractmethod
    async def add_batch(self, memories: List[Memory]) -> List[Memory]:
        """Add a batch of memories to the store.
        
        Args:
            memories: Memuries to add
            
        Returns:
            List[Memory]: Added memories with IDs and any other fields populated
        """
        ...
    
    @abc.abstractmethod
    async def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID.
        
        Args:
            memory_id: ID of the memory to get
            
        Returns:
            Optional[Memory]: Memory if found, None otherwise
        """
        ...
    
    @abc.abstractmethod
    async def update(self, memory: Memory) -> Memory:
        """Update a memory in the store.
        
        Args:
            memory: Memory to update
            
        Returns:
            Memory: Updated memory
        """
        ...
    
    @abc.abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            bool: True if the memory was deleted, False otherwise
        """
        ...
    
    @abc.abstractmethod
    async def search(self, query: SearchQuery) -> SearchResult:
        """Search for memories.
        
        Args:
            query: Search query
            
        Returns:
            SearchResult: Search results
        """
        ...
    
    @abc.abstractmethod
    async def search_by_vector(
        self, 
        vector: List[float], 
        category: Optional[MemoryCategory] = None,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredMemory]:
        """Search for memories by vector similarity.
        
        Args:
            vector: Query vector
            category: Optional category filter
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
            
        Returns:
            List[ScoredMemory]: Scored memories
        """
        ...
    
    @abc.abstractmethod
    async def search_by_text(
        self, 
        text: str,
        category: Optional[MemoryCategory] = None,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredMemory]:
        """Search for memories by text.
        
        This method embeds the query text and then searches by vector.
        
        Args:
            text: Query text
            category: Optional category filter
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
            
        Returns:
            List[ScoredMemory]: Scored memories
        """
        ...
    
    @abc.abstractmethod
    async def count(
        self, 
        category: Optional[MemoryCategory] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Count memories matching the given filters.
        
        Args:
            category: Optional category filter
            metadata_filters: Optional metadata filters
            
        Returns:
            int: Number of matching memories
        """
        ...


class LLMService(Protocol):
    """Interface for LLM services."""
    
    @abc.abstractmethod
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
        ...
    
    @abc.abstractmethod
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
        ...
    
    @abc.abstractmethod
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
        ...
    
    @abc.abstractmethod
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
        ...


class ClassifierService(Protocol):
    """Interface for classifying text into categories."""
    
    @abc.abstractmethod
    async def classify(self, text: str) -> Dict[MemoryCategory, float]:
        """Classify text into memory categories.
        
        Args:
            text: Text to classify
            
        Returns:
            Dict[MemoryCategory, float]: Categories with confidence scores
        """
        ...
    
    @abc.abstractmethod
    async def train(self, texts: List[str], categories: List[MemoryCategory]) -> None:
        """Train the classifier on new examples.
        
        Args:
            texts: Training texts
            categories: Corresponding categories
        """
        ...


class FeedbackService(Protocol):
    """Interface for handling classifier feedback."""
    
    @abc.abstractmethod
    async def log_feedback(
        self, 
        text: str, 
        category: MemoryCategory,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log feedback for the classifier.
        
        Args:
            text: Text that was classified
            category: Correct category
            metadata: Optional metadata
        """
        ...
    
    @abc.abstractmethod
    async def retrain_classifier(self) -> None:
        """Retrain the classifier using collected feedback."""
        ...


class RerankingService(Protocol):
    """Interface for reranking search results."""
    
    @abc.abstractmethod
    async def rerank(
        self, 
        query: str, 
        candidates: List[ScoredMemory],
        top_k: int = 5,
    ) -> List[ScoredMemory]:
        """Rerank search results.
        
        Args:
            query: Search query
            candidates: Initial search results
            top_k: Number of results to return
            
        Returns:
            List[ScoredMemory]: Reranked search results
        """
        ...


class VectorIndex(Protocol):
    """Interface for in-memory vector indices."""
    
    @abc.abstractmethod
    def add(self, id: str, vector: List[float]) -> None:
        """Add a vector to the index.
        
        Args:
            id: Vector ID
            vector: Vector to add
        """
        ...
    
    @abc.abstractmethod
    def search(self, query: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query: Query vector
            top_k: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Search results with IDs and distances
        """
        ...
    
    @abc.abstractmethod
    def delete(self, id: str) -> None:
        """Delete a vector from the index.
        
        Args:
            id: ID of the vector to delete
        """
        ...
    
    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Save the index to disk.
        
        Args:
            path: Path to save the index to
        """
        ...
    
    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load the index from disk.
        
        Args:
            path: Path to load the index from
        """
        ...
    
    @abc.abstractmethod
    def get_size(self) -> int:
        """Get the number of vectors in the index.
        
        Returns:
            int: Number of vectors
        """
        ... 