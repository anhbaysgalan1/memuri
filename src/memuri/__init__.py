"""
Memuri - A high-performance, pluggable conversational memory services SDK.

This package provides tools for managing short and long-term conversational memory
with pluggable vector databases, embedding services, and LLM providers.
"""

__version__ = "0.1.0"

# Export key classes for easier imports
from memuri.sdk.client import Memuri
from memuri.domain.models import ChatMessage, Document, Memory, MemoryCategory
from memuri.core.config import (
    MemuriSettings, 
    EmbeddingSettings, 
    LLMSettings, 
    MemoryRuleSettings,
    RerankSettings,
)

__all__ = [
    "Memuri", 
    "ChatMessage", 
    "Document", 
    "Memory",
    "MemoryCategory",
    "MemuriSettings",
    "EmbeddingSettings",
    "LLMSettings",
    "MemoryRuleSettings",
    "RerankSettings",
] 