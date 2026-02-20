# logic_core.py

# --- 1. CRISIS SAFETY NET ---
CRISIS_KEYWORDS = [
    "kill myself", "killing myself", "kill me", "suicide", "suicidal",
    "end my life", "end it all", "want to die", "better off dead",
    "hurt myself", "hurting myself", "harm myself", "cut myself", 
    "cutting myself", "overdose", "shoot myself", "hang myself",
    "sleep and never wake", "never wake up", "don't want to wake up"
]

MOOD_COLORS = {
    "Crisis": "#FF5252", 
    "Depression": "#90CAF9", 
    "Anxiety": "#CE93D8",
    "Stress": "#FFAB91", 
    "Loneliness": "#9FA8DA", 
    "Happy": "#FFD700",
    "Calm": "#B2DFDB", 
    "Normal": "#E0E0E0"
}

def check_safety(text):
    clean_text = text.lower()
    for word in CRISIS_KEYWORDS:
        if word in clean_text:
            return True, "I'm hearing that you are in pain. Please know you are not alone."
    return False, ""

def get_feedback_and_color(mood, username="Friend"):
    # Ensure mood is valid, default to Normal
    if mood not in MOOD_COLORS:
        mood = "Normal"

    feedback_map = {
        "Happy": f"That's wonderful, {username}! Keep holding onto this feeling.",
        "Calm": f"It is good to feel at peace, {username}.",
        "Depression": f"I hear you, {username}. It's okay to let those feelings out.",
        "Anxiety": f"Take a deep breath, {username}. I am here with you.",
        "Stress": f"You are carrying a lot, {username}. Let's take it one step at a time.",
        "Loneliness": f"I am here, {username}. You are connected to me right now.",
        "Normal": f"Got it, {username}. How was your day otherwise?",
        "Crisis": "I'm detecting that you might be in distress. Please reach out to your safety contact."
    }

    return {
        "mood": mood,
        "feedback": feedback_map[mood],
        "color_code": MOOD_COLORS[mood]
    }