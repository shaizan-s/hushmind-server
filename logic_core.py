# logic_core.py
import re

# --- 1. CONFIGURATION ---
# Fallback keywords if AI fails
CRISIS_KEYWORDS = [
    "kill myself", "suicide", "die", "end it", "hurt myself", 
    "cutting", "overdose", "dead", "want to die", "sleep and never wake",
    "never wake up"
]

MOOD_COLORS = {
    "Crisis": "#FF5252",    # Red
    "Depression": "#90CAF9",# Blue
    "Anxiety": "#CE93D8",   # Purple
    "Stress": "#FFAB91",    # Orange
    "Loneliness": "#9FA8DA",# Indigo
    "Happy": "#FFD700",     # Gold
    "Calm": "#B2DFDB",      # Teal
    "Normal": "#E0E0E0"     # Grey
}

# --- 2. SAFETY FUNCTIONS ---
def check_safety_keywords(text):
    """
    Fastest check: Does it contain a known danger phrase?
    """
    clean_text = text.lower()
    for word in CRISIS_KEYWORDS:
        if word in clean_text:
            return True, "I'm hearing that you are in pain. Please know you are not alone."
    return False, ""

# --- 3. THE CONFLICT RESOLUTION LAYER ---
def resolve_mood(text, suicide_model, mood_model, username="Friend"):
    """
    The Master Logic:
    1. Check Keywords (Fastest)
    2. Check Suicide Model (The Guard Dog)
    3. Check Mood Model (The Therapist) - ONLY if safe
    """
    
    # A. Keyword Check
    is_keyword_danger, msg = check_safety_keywords(text)
    if is_keyword_danger:
        return {"mood": "Crisis", "feedback": msg, "color_code": MOOD_COLORS["Crisis"]}

    # B. AI Guard Dog Check (Suicide Model)
    if suicide_model:
        # We predict using the binary safety model
        safety_label = suicide_model.predict([text])[0]
        
        # 🛑 TRUMP CARD RULE: If Guard Dog says Crisis, we STOP here.
        if safety_label == "Crisis":
            print(f"🚨 SUICIDE MODEL TRIGGERED: {text}")
            return {
                "mood": "Crisis", 
                "feedback": "I'm detecting that you might be in distress. Please reach out to your safety contact.", 
                "color_code": MOOD_COLORS["Crisis"]
            }

    # C. AI Therapist Check (Mood Model)
    # If we reached here, the user is SAFE. Now we check feelings.
    final_mood = "Normal" # Default
    if mood_model:
        final_mood = mood_model.predict([text])[0]

    # D. Generate Feedback
    feedback_map = {
        "Happy": f"That's wonderful, {username}! Keep holding onto this feeling.",
        "Calm": f"It is good to feel at peace, {username}.",
        "Depression": f"I hear you, {username}. You are not alone in this darkness.",
        "Anxiety": f"Take a deep breath, {username}. I am here with you.",
        "Stress": f"It sounds heavy, {username}. Let's take it one step at a time.",
        "Loneliness": f"I am here, {username}. You are connected to me right now.",
        "Normal": f"Got it, {username}. How was your day otherwise?"
    }

    return {
        "mood": final_mood,
        "feedback": feedback_map.get(final_mood, f"I am listening, {username}."),
        "color_code": MOOD_COLORS.get(final_mood, "#E0E0E0")
    }