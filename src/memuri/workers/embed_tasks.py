"""
Embedding tasks for Celery workers.

This module provides Celery tasks for asynchronous embedding generation.
"""

from typing import List, Dict, Any, Optional
import logging

from celery import shared_task
from memuri.factory import EmbedderFactory
from memuri.core.config import get_settings
from memuri.core.text_utils import normalize_text_for_embedding

logger = logging.getLogger(__name__)


@shared_task(
    name="memuri.workers.embed_tasks.embed_text",
    bind=True,
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def embed_text(self, text: str) -> List[float]:
    """
    Generate an embedding for a text string.

    Args:
        text: The text to embed

    Returns:
        List[float]: The embedding vector
    """
    # Clean and normalize the text
    cleaned_text = normalize_text_for_embedding(text)

    # Get the settings and create the embedder
    settings = get_settings()
    embedding_service = EmbedderFactory.create(
        provider=settings.embedding.provider,
        settings=settings.embedding
    )

    # Generate the embedding
    try:
        result = embedding_service.embed(cleaned_text)
        logger.info(f"Generated embedding of length {len(result)}")
        return result
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise self.retry(exc=e)


@shared_task(
    name="memuri.workers.embed_tasks.embed_batch",
    bind=True,
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def embed_batch(self, texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of text strings.

    Args:
        texts: List of text strings to embed

    Returns:
        List[List[float]]: List of embedding vectors
    """
    # Clean and normalize the texts
    cleaned_texts = [normalize_text_for_embedding(text) for text in texts]

    # Get the settings and create the embedder
    settings = get_settings()
    embedding_service = EmbedderFactory.create(
        provider=settings.embedding.provider,
        settings=settings.embedding
    )

    # Generate the embeddings
    try:
        results = embedding_service.embed_batch(cleaned_texts)
        logger.info(f"Generated {len(results)} embeddings")
        return results
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise self.retry(exc=e)


@shared_task(
    name="memuri.workers.embed_tasks.embed_document",
    bind=True,
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def embed_document(
    self,
    document: Dict[str, Any],
    text_field: str = "content",
    metadata_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate an embedding for a document and store it in the document.

    Args:
        document: Document dictionary with content
        text_field: Key for the text content in the document
        metadata_fields: Optional keys for metadata to include in embedding text

    Returns:
        Dict[str, Any]: Document with added embedding field
    """
    # Extract text from document
    text = document.get(text_field, "")
    
    # Optionally add metadata fields to the text
    if metadata_fields:
        for field in metadata_fields:
            if field in document and field != text_field:
                field_value = document.get(field)
                if field_value:
                    text += f" {field}: {field_value}"

    # Generate embedding
    embedding = embed_text.apply(args=[text]).get()
    
    # Add embedding to document
    document["embedding"] = embedding
    return document 