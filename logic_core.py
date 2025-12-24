# logic_core.py
import re

# --- 1. CONFIGURATION ---
CRISIS_KEYWORDS = [
    "kill myself", "suicide", "die", "end it", "hurt myself", 
    "cutting", "overdose", "dead", "want to die"
]

MOOD_COLORS = {
    "Crisis": "0xFFFF0000", # Red
    "Sad": "0xFF6A85B6",    # Blue-Grey
    "Happy": "0xFFF6D365",  # Yellow/Gold
    "Calm": "0xFFA8E063",   # Green
    "Lonely": "0xFF9D50BB", # Purple
    "Angry": "0xFFEB5757"   # Red-Orange
}

# --- 2. SAFETY FUNCTIONS ---
def check_safety(text):
    """
    Returns (is_crisis: bool, message: str)
    """
    clean_text = text.lower()
    for word in CRISIS_KEYWORDS:
        if word in clean_text:
            return True, "I'm hearing that you are in pain. Please know you are not alone. Please contact a crisis helpline immediately."
    return False, ""

# --- 3. ANALYSIS FUNCTIONS ---
def resolve_mood(text, ml_prediction, username="Friend"):
    """
    Combines ML guess with Safety checks and returns formatted feedback.
    """
    # 1. Check Safety First (Override ML)
    is_crisis, crisis_msg = check_safety(text)
    if is_crisis:
        return {
            "mood": "Crisis",
            "feedback": crisis_msg,
            "color_code": MOOD_COLORS["Crisis"]
        }

    # 2. Manual Logic Overrides (for common errors)
    clean_text = text.lower().strip()
    if clean_text in ["sad", "i am sad", "depressed"]:
        final_mood = "Sad"
    elif clean_text in ["happy", "good", "great"]:
        final_mood = "Happy"
    else:
        final_mood = ml_prediction # Trust the ML model

    # 3. Generate Feedback
    feedback_map = {
        "Sad": f"I hear you, {username}. It's okay to let those feelings out.",
        "Lonely": f"I'm here with you, {username}. Loneliness is a sign we need connection.",
        "Happy": f"That's wonderful, {username}! Hold onto this feeling.",
        "Calm": "It sounds like things are steady. That is a good place to be.",
        "Angry": f"Take a deep breath, {username}. It's safe to be angry here."
    }

    return {
        "mood": final_mood,
        "feedback": feedback_map.get(final_mood, "I am listening."),
        "color_code": MOOD_COLORS.get(final_mood, "0xFFFFFFFF")
    }

def extract_themes(text):
    """
    Simple keyword extraction for the journal.
    """
    themes = []
    text = text.lower()
    if "work" in text or "job" in text: themes.append("Work")
    if "family" in text or "mom" in text or "dad" in text: themes.append("Family")
    if "sleep" in text or "tired" in text: themes.append("Health")
    if "love" in text or "relationship" in text: themes.append("Relationships")
    
    if not themes:
        themes.append("General Reflection")
        
    return themes