"""Keyword-based classifier for memories."""

from typing import Dict, List, Optional

from memuri.core.categories import CATEGORY_SUBCATEGORY_MAP, get_parent_category
from memuri.domain.models import MemoryCategory


class KeywordClassifier:
    """Simple keyword-based classifier for memory categorization."""
    
    def __init__(self):
        """Initialize the keyword classifier with default keywords."""
        # Define keywords for each category and subcategory
        self.category_keywords = {
            # Profile Information
            MemoryCategory.PROFILE_INFORMATION: ["profile", "about me", "personal", "bio", "identity"],
            MemoryCategory.PERSONAL_INFORMATION: ["name", "age", "birthday", "gender", "address", "contact"],
            MemoryCategory.DEMOGRAPHICS: ["age", "gender", "location", "demographic", "country", "city"],
            MemoryCategory.IDENTITY_TRAITS: ["personality", "traits", "character", "introvert", "extrovert", "values"],
            MemoryCategory.PERSONAL_DETAILS: ["details", "background", "personal info", "identifying"],
            
            # Preferences
            MemoryCategory.PREFERENCES: ["like", "dislike", "prefer", "favorite", "hate", "enjoy"],
            MemoryCategory.FAVORITE_TOPICS: ["topics", "interest", "subject", "theme", "field"],
            MemoryCategory.COMMUNICATION_STYLE: ["communication", "talk", "chat", "speak", "writing", "style"],
            MemoryCategory.MEDIA_PREFERENCES: ["media", "movies", "music", "books", "shows", "games"],
            
            # Goals & Aspirations
            MemoryCategory.GOALS_ASPIRATIONS: ["goal", "aspiration", "ambition", "dream", "future", "aim"],
            MemoryCategory.CAREER_GOALS: ["career", "job", "profession", "work", "employment"],
            MemoryCategory.PERSONAL_GOALS: ["personal goal", "self-improvement", "growth", "skill", "development"],
            MemoryCategory.PROJECT_ASPIRATIONS: ["project", "initiative", "venture", "startup", "creation"],
            
            # Routines & Habits
            MemoryCategory.ROUTINES_HABITS: ["routine", "habit", "schedule", "practice", "ritual", "regular"],
            MemoryCategory.DAILY_ROUTINES: ["daily", "morning", "evening", "night", "day", "schedule"],
            MemoryCategory.HEALTH_HABITS: ["health", "exercise", "fitness", "diet", "nutrition", "sleep"],
            MemoryCategory.PRODUCTIVITY_HABITS: ["productivity", "work", "efficiency", "focus", "task", "time"],
            
            # Events & Appointments
            MemoryCategory.EVENTS_APPOINTMENTS: ["event", "appointment", "meeting", "schedule", "calendar"],
            MemoryCategory.CALENDAR_EVENTS: ["calendar", "schedule", "appointment", "meeting", "upcoming"],
            MemoryCategory.MILESTONES: ["milestone", "achievement", "celebration", "significant", "important"],
            MemoryCategory.TRAVEL_PLANS: ["travel", "trip", "vacation", "journey", "visit", "destination"],
            
            # Projects & Tasks
            MemoryCategory.PROJECTS_TASKS: ["project", "task", "work", "assignment", "job", "duty"],
            MemoryCategory.ACTIVE_PROJECTS: ["active", "current", "ongoing", "project", "initiative", "job"],
            MemoryCategory.TO_DO_ITEMS: ["todo", "to-do", "task", "reminder", "assignment", "checklist"],
            MemoryCategory.BACKLOG_ITEMS: ["backlog", "future", "pending", "later", "queue", "list"],
            
            # Health & Wellness
            MemoryCategory.HEALTH_WELLNESS: ["health", "wellness", "wellbeing", "fitness", "medical", "condition"],
            MemoryCategory.MEDICAL_CONDITIONS: ["medical", "condition", "disease", "illness", "health issue", "diagnosis"],
            MemoryCategory.DIETARY_PREFERENCES: ["diet", "food", "nutrition", "eating", "dietary", "preference"],
            MemoryCategory.WELLNESS_METRICS: ["metric", "measurement", "tracking", "stats", "data", "health data"],
            
            # Social Relationships
            MemoryCategory.SOCIAL_RELATIONSHIPS: ["social", "relationship", "connection", "network", "contact", "people"],
            MemoryCategory.FAMILY_MEMBERS: ["family", "parent", "child", "sibling", "relative", "spouse"],
            MemoryCategory.FRIENDS_NETWORK: ["friend", "buddy", "pal", "acquaintance", "social circle", "peer"],
            MemoryCategory.PROFESSIONAL_CONTACTS: ["professional", "colleague", "coworker", "client", "business", "networking"],
            
            # Skills & Knowledge
            MemoryCategory.SKILLS_KNOWLEDGE: ["skill", "knowledge", "expertise", "ability", "education", "know-how"],
            MemoryCategory.TECHNICAL_SKILLS: ["technical", "coding", "programming", "tech", "software", "hardware"],
            MemoryCategory.LANGUAGES_SPOKEN: ["language", "spoken", "fluent", "native", "bilingual", "speak"],
            MemoryCategory.CERTIFICATIONS: ["certification", "certificate", "credential", "qualification", "degree", "diploma"],
            
            # Experiences & Memories
            MemoryCategory.EXPERIENCES_MEMORIES: ["experience", "memory", "recollection", "moment", "past", "remember"],
            MemoryCategory.TRAVEL_EXPERIENCES: ["travel", "trip", "journey", "visit", "destination", "tourism"],
            MemoryCategory.EDUCATIONAL_BACKGROUND: ["education", "school", "college", "university", "degree", "academic"],
            MemoryCategory.NOTABLE_LIFE_EVENTS: ["life event", "milestone", "significant", "memorable", "important", "life-changing"],
            
            # Feedback & Opinions
            MemoryCategory.FEEDBACK_OPINIONS: ["feedback", "opinion", "thought", "view", "stance", "belief"],
            MemoryCategory.PRODUCT_FEEDBACK: ["product", "service", "app", "software", "review", "rating"],
            MemoryCategory.PERSONAL_OPINIONS: ["opinion", "belief", "perspective", "viewpoint", "stance", "position"],
            MemoryCategory.SUGGESTIONS: ["suggestion", "recommend", "advice", "idea", "proposal", "tip"],
            
            # Financial Info
            MemoryCategory.FINANCIAL_INFO: ["financial", "money", "finance", "budget", "expense", "income"],
            MemoryCategory.BUDGET_GOALS: ["budget", "financial goal", "saving", "expense", "target", "money goal"],
            MemoryCategory.EXPENSES_LOG: ["expense", "spending", "cost", "purchase", "payment", "transaction"],
            MemoryCategory.INVESTMENT_PREFERENCES: ["investment", "investing", "stock", "fund", "portfolio", "asset"],
            
            # Media Content
            MemoryCategory.MEDIA_CONTENT: ["media", "content", "book", "article", "video", "audio"],
            MemoryCategory.BOOKS_READ: ["book", "read", "novel", "textbook", "reading", "literature"],
            MemoryCategory.ARTICLES_CONSUMED: ["article", "blog", "post", "news", "publication", "reading"],
            MemoryCategory.MULTIMEDIA_ENGAGEMENT: ["video", "movie", "film", "audio", "podcast", "show"],
            
            # Contextual Metadata
            MemoryCategory.CONTEXTUAL_METADATA: ["context", "metadata", "data", "information", "setting", "environment"],
            MemoryCategory.DEVICE_INFO: ["device", "computer", "phone", "hardware", "system", "platform"],
            MemoryCategory.SESSION_PREFERENCES: ["session", "setting", "preference", "config", "configuration", "setup"],
            MemoryCategory.LOCATION_HISTORY: ["location", "place", "visit", "travel", "where", "venue"],
            
            # Miscellaneous
            MemoryCategory.MISCELLANEOUS: ["misc", "other", "general", "various", "assorted", "diverse"],
            MemoryCategory.MISC: ["misc", "other", "general", "various", "assorted", "diverse"],
        }
    
    async def classify(self, text: str) -> Dict[MemoryCategory, float]:
        """Classify text into memory categories.
        
        Args:
            text: Text to classify
            
        Returns:
            Dict[MemoryCategory, float]: Categories with confidence scores
        """
        # Convert text to lowercase for case-insensitive matching
        text = text.lower()
        
        # Initialize scores for all categories
        scores: Dict[MemoryCategory, float] = {
            category: 0.0 for category in MemoryCategory
        }
        
        # Set a minimum score for MISCELLANEOUS
        scores[MemoryCategory.MISCELLANEOUS] = 0.1
        
        # Count keyword occurrences for each category
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    scores[category] += 0.15  # Increase score for each keyword found
        
        # Boost parent category scores based on subcategory scores
        for subcategory, score in scores.items():
            if score > 0:
                subcategory_str = subcategory.value
                parent_category_str = get_parent_category(subcategory_str)
                
                if parent_category_str != subcategory_str:
                    try:
                        parent_category = MemoryCategory(parent_category_str)
                        scores[parent_category] += score * 0.5
                    except ValueError:
                        pass  # Parent category not in MemoryCategory enum
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for category in scores:
                scores[category] /= total
        
        return scores
    
    async def train(self, texts: List[str], categories: List[MemoryCategory]) -> None:
        """Train the classifier on new examples.
        
        This is a no-op for the keyword classifier.
        
        Args:
            texts: Training texts
            categories: Corresponding categories
        """
        # Keyword classifier doesn't learn, so this is a no-op
        pass 