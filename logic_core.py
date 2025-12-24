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
    Combines ML guess with Safety checks and keyword overrides.
    """
    clean_text = text.lower()

    # 1. Check Safety First (Highest Priority)
    is_crisis, crisis_msg = check_safety(text)
    if is_crisis:
        return {
            "mood": "Crisis",
            "feedback": crisis_msg,
            "color_code": MOOD_COLORS["Crisis"]
        }

    # 2. Positive Keyword Override (Fixes the "Work Promotion" bug)
    # If the user says "happy", "proud", or "excited", we FORCE the mood to Happy.
    positive_keywords = ["happy", "excited", "proud", "great", "joy", "awesome", "good", "promotion", "love"]
    if any(word in clean_text for word in positive_keywords):
        final_mood = "Happy"
        
    # 3. Manual Logic Overrides for negatives (if ML misses them)
    elif "sad" in clean_text or "depressed" in clean_text or "crying" in clean_text:
        final_mood = "Sad"
    elif "anxious" in clean_text or "nervous" in clean_text:
        final_mood = "Stress"
        
    # 4. If no keywords found, trust the ML model
    else:
        final_mood = ml_prediction 

    # 5. Generate Feedback
    feedback_map = {
        "Sad": f"I hear you, {username}. It's okay to let those feelings out.",
        "Lonely": f"I'm here with you, {username}. Loneliness is a sign we need connection.",
        "Happy": f"That's wonderful, {username}! Hold onto this feeling.",
        "Calm": "It sounds like things are steady. That is a good place to be.",
        "Angry": f"Take a deep breath, {username}. It's safe to be angry here.",
        "Stress": f"You're carrying a lot, {username}. Remember to breathe.",
        "Normal": f"Day by day, {username}. That's how we do it."
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