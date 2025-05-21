"""Category and subcategory mapping for memory classification."""

from typing import Dict, List

# Define the mapping of main categories to their subcategories
CATEGORY_SUBCATEGORY_MAP: Dict[str, List[str]] = {
    "profile_information": [
        "personal_information", "demographics", "identity_traits", 
        "personal_details"  # Adding alternative name
    ],
    "preferences": [
        "favorite_topics", "communication_style", "media_preferences"
    ],
    "goals_aspirations": [
        "career_goals", "personal_goals", "project_aspirations"
    ],
    "routines_habits": [
        "daily_routines", "health_habits", "productivity_habits"
    ],
    "events_appointments": [
        "calendar_events", "milestones", "travel_plans"
    ],
    "projects_tasks": [
        "active_projects", "to_do_items", "backlog_items"
    ],
    "health_wellness": [
        "medical_conditions", "dietary_preferences", "wellness_metrics"
    ],
    "social_relationships": [
        "family_members", "friends_network", "professional_contacts"
    ],
    "skills_knowledge": [
        "technical_skills", "languages_spoken", "certifications"
    ],
    "experiences_memories": [
        "travel_experiences", "educational_background", "notable_life_events"
    ],
    "feedback_opinions": [
        "product_feedback", "personal_opinions", "suggestions"
    ],
    "financial_info": [
        "budget_goals", "expenses_log", "investment_preferences"
    ],
    "media_content": [
        "books_read", "articles_consumed", "multimedia_engagement"
    ],
    "contextual_metadata": [
        "device_info", "session_preferences", "location_history"
    ],
    "miscellaneous": ["misc"]
}

# Map for legacy categories
LEGACY_CATEGORY_MAP = {
    "PERSONAL": "profile_information",
    "TASK": "projects_tasks",
    "QUESTION": "feedback_opinions",
    "EMOTION": "profile_information",
    "DECISION": "feedback_opinions", 
    "TODO": "projects_tasks",
    "FACT": "skills_knowledge",
    "GENERAL": "miscellaneous"
}

def get_parent_category(subcategory: str) -> str:
    """Get the parent category for a given subcategory.
    
    Args:
        subcategory: The subcategory to lookup
        
    Returns:
        str: The parent category, or the subcategory itself if it's a main category
    """
    # Check if it's a main category
    if subcategory in CATEGORY_SUBCATEGORY_MAP:
        return subcategory
        
    # Search for the subcategory in all categories
    for category, subcategories in CATEGORY_SUBCATEGORY_MAP.items():
        if subcategory in subcategories:
            return category
    
    # Handle legacy categories
    if subcategory in LEGACY_CATEGORY_MAP:
        return LEGACY_CATEGORY_MAP[subcategory]
            
    # Default to miscellaneous if not found
    return "miscellaneous" 