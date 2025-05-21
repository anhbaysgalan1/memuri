"""Domain models for the memuri SDK."""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a message in a chat conversation."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Represents a chat message in the memory system."""
    
    id: Optional[str] = Field(None, description="Unique identifier for the message")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now, 
        description="Timestamp when the message was created"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Arbitrary metadata associated with the message"
    )
    
    @property
    def age_seconds(self) -> float:
        """Get the age of the message in seconds.
        
        Returns:
            float: Age in seconds
        """
        return (datetime.datetime.now() - self.created_at).total_seconds()


class DocumentType(str, Enum):
    """Type of document in the memory system."""
    
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    URL = "url"
    PDF = "pdf"


class Document(BaseModel):
    """Represents a document in the memory system."""
    
    id: Optional[str] = Field(None, description="Unique identifier for the document")
    content: str = Field(..., description="Content of the document")
    doc_type: DocumentType = Field(DocumentType.TEXT, description="Type of document")
    title: Optional[str] = Field(None, description="Title of the document")
    source: Optional[str] = Field(None, description="Source of the document")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp when the document was created"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata associated with the document"
    )
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the document content")


class DocumentChunk(BaseModel):
    """Represents a chunk of a document in the memory system."""
    
    id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="ID of the parent document")
    content: str = Field(..., description="Content of the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata associated with the chunk"
    )


class MemoryCategory(str, Enum):
    """Category of memory in the system."""
    
    # Main categories
    PROFILE_INFORMATION = "profile_information"
    PREFERENCES = "preferences"
    GOALS_ASPIRATIONS = "goals_aspirations"
    ROUTINES_HABITS = "routines_habits"
    EVENTS_APPOINTMENTS = "events_appointments"
    PROJECTS_TASKS = "projects_tasks"
    HEALTH_WELLNESS = "health_wellness"
    SOCIAL_RELATIONSHIPS = "social_relationships"
    SKILLS_KNOWLEDGE = "skills_knowledge"
    EXPERIENCES_MEMORIES = "experiences_memories"
    FEEDBACK_OPINIONS = "feedback_opinions"
    FINANCIAL_INFO = "financial_info"
    MEDIA_CONTENT = "media_content"
    CONTEXTUAL_METADATA = "contextual_metadata"
    
    # Subcategories - Profile Information
    PERSONAL_INFORMATION = "personal_information"
    DEMOGRAPHICS = "demographics"
    IDENTITY_TRAITS = "identity_traits"
    PERSONAL_DETAILS = "personal_details"
    
    # Subcategories - Preferences
    FAVORITE_TOPICS = "favorite_topics"
    COMMUNICATION_STYLE = "communication_style"
    MEDIA_PREFERENCES = "media_preferences"
    
    # Subcategories - Goals & Aspirations
    CAREER_GOALS = "career_goals"
    PERSONAL_GOALS = "personal_goals"
    PROJECT_ASPIRATIONS = "project_aspirations"
    
    # Subcategories - Routines & Habits
    DAILY_ROUTINES = "daily_routines"
    HEALTH_HABITS = "health_habits"
    PRODUCTIVITY_HABITS = "productivity_habits"
    
    # Subcategories - Events & Appointments
    CALENDAR_EVENTS = "calendar_events"
    MILESTONES = "milestones"
    TRAVEL_PLANS = "travel_plans"
    
    # Subcategories - Projects & Tasks
    ACTIVE_PROJECTS = "active_projects"
    TO_DO_ITEMS = "to_do_items"
    BACKLOG_ITEMS = "backlog_items"
    
    # Subcategories - Health & Wellness
    MEDICAL_CONDITIONS = "medical_conditions"
    DIETARY_PREFERENCES = "dietary_preferences"
    WELLNESS_METRICS = "wellness_metrics"
    
    # Subcategories - Social Relationships
    FAMILY_MEMBERS = "family_members"
    FRIENDS_NETWORK = "friends_network"
    PROFESSIONAL_CONTACTS = "professional_contacts"
    
    # Subcategories - Skills & Knowledge
    TECHNICAL_SKILLS = "technical_skills"
    LANGUAGES_SPOKEN = "languages_spoken"
    CERTIFICATIONS = "certifications"
    
    # Subcategories - Experiences & Memories
    TRAVEL_EXPERIENCES = "travel_experiences"
    EDUCATIONAL_BACKGROUND = "educational_background"
    NOTABLE_LIFE_EVENTS = "notable_life_events"
    
    # Subcategories - Feedback & Opinions
    PRODUCT_FEEDBACK = "product_feedback"
    PERSONAL_OPINIONS = "personal_opinions"
    SUGGESTIONS = "suggestions"
    
    # Subcategories - Financial Info
    BUDGET_GOALS = "budget_goals"
    EXPENSES_LOG = "expenses_log"
    INVESTMENT_PREFERENCES = "investment_preferences"
    
    # Subcategories - Media Content
    BOOKS_READ = "books_read"
    ARTICLES_CONSUMED = "articles_consumed"
    MULTIMEDIA_ENGAGEMENT = "multimedia_engagement"
    
    # Subcategories - Contextual Metadata
    DEVICE_INFO = "device_info"
    SESSION_PREFERENCES = "session_preferences"
    LOCATION_HISTORY = "location_history"
    
    # Miscellaneous
    MISCELLANEOUS = "miscellaneous"
    MISC = "misc"
    
    # Legacy categories for backward compatibility
    PERSONAL = "PERSONAL"
    TASK = "TASK"
    QUESTION = "QUESTION"
    EMOTION = "EMOTION"
    DECISION = "DECISION"
    TODO = "TODO"
    FACT = "FACT"
    GENERAL = "GENERAL"


class MemorySource(str, Enum):
    """Source of a memory entry."""
    
    USER = "user"
    AUTO = "auto"
    DERIVED = "derived"
    SYSTEM = "system"


class Memory(BaseModel):
    """Represents a memory entry in the system."""
    
    id: Optional[str] = Field(None, description="Unique identifier for the memory")
    content: str = Field(..., description="Content of the memory")
    category: MemoryCategory = Field(
        MemoryCategory.GENERAL, 
        description="Category of the memory"
    )
    source: MemorySource = Field(
        MemorySource.USER, 
        description="Source of the memory"
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp when the memory was created"
    )
    embedding: Optional[List[float]] = Field(
        None, 
        description="Vector embedding of the memory content"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata associated with the memory"
    )
    
    class Config:
        """Pydantic config."""
        
        json_encoders = {
            datetime.datetime: lambda v: v.isoformat(),
        }


class ScoredMemory(BaseModel):
    """A memory with a relevance score."""
    
    memory: Memory = Field(..., description="The memory")
    score: float = Field(..., description="Relevance score")


class EmbeddingResponse(BaseModel):
    """Response from an embedding service."""
    
    embeddings: List[List[float]] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used for embedding")
    dimensions: int = Field(..., description="Dimensions of the embeddings")
    tokens: int = Field(..., description="Total tokens used")


class SearchQuery(BaseModel):
    """A query for searching memories."""
    
    query: str = Field(..., description="Query string")
    category: Optional[MemoryCategory] = Field(
        None, 
        description="Filter by category"
    )
    top_k: int = Field(5, description="Number of results to return")
    rerank: bool = Field(True, description="Whether to rerank results")
    metadata_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filters to apply to metadata"
    )


class SearchResult(BaseModel):
    """Result of a memory search."""
    
    memories: List[ScoredMemory] = Field(..., description="Scored memories")
    query: str = Field(..., description="Original query")
    total_found: int = Field(..., description="Total number of matching memories")
    search_time: float = Field(..., description="Search time in seconds")
    reranked: bool = Field(False, description="Whether results were reranked") 