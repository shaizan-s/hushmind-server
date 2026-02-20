import re

# --- 1. THE FOOLPROOF DICTIONARY ---
EXPLICIT_MOODS = {
    "Depression": ["sad", "depressed", "depressing", "crying", "miserable", "unhappy", "hopeless", "down"],
    "Anxiety": ["anxious", "anxiety", "nervous", "panic", "scared", "worried", "fear"],
    "Stress": ["stress", "stressed", "overwhelmed", "pressure", "exhausted", "tired"],
    "Loneliness": ["lonely", "alone", "isolated", "nobody", "ignored"],
    "Happy": ["happy", "joy", "excited", "great", "awesome", "good", "glad"],
    "Calm": ["calm", "peace", "relax", "chill", "okay", "alright", "fine"]
}

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

# --- 2. LOGIC FUNCTIONS ---
def check_safety(text):
    clean_text = text.lower()
    for word in CRISIS_KEYWORDS:
        if word in clean_text:
            return True, "I'm hearing that you are in pain. Please know you are not alone."
    return False, ""

def resolve_mood(text, ml_prediction, username="Friend"):
    # 1. CRISIS CHECK (Highest Priority)
    is_crisis, msg = check_safety(text)
    if is_crisis:
        return {"mood": "Crisis", "feedback": msg, "color_code": MOOD_COLORS["Crisis"]}

    # 2. DICTIONARY CHECK (Bypasses AI completely for exact words)
    clean_text = text.lower()
    words = re.findall(r'\b\w+\b', clean_text)
    
    final_mood = None
    for mood, keywords in EXPLICIT_MOODS.items():
        if any(kw in words for kw in keywords):
            final_mood = mood
            break
    
    # 3. FALLBACK TO AI (If no exact words used)
    if not final_mood:
        # Convert old AI labels to the new app labels just in case
        if ml_prediction == "Sad": final_mood = "Depression"
        elif ml_prediction == "Lonely": final_mood = "Loneliness"
        elif ml_prediction == "Angry": final_mood = "Stress"
        else: final_mood = ml_prediction
        
    # Ensure it's a valid mood, otherwise Normal
    if final_mood not in MOOD_COLORS:
        final_mood = "Normal"

    # 4. FEEDBACK
    feedback_map = {
        "Happy": f"That's wonderful, {username}! Keep holding onto this feeling.",
        "Calm": f"It is good to feel at peace, {username}.",
        "Depression": f"I hear you, {username}. It's okay to let those feelings out.",
        "Anxiety": f"Take a deep breath, {username}. I am here with you.",
        "Stress": f"You are carrying a lot, {username}. Let's take it one step at a time.",
        "Loneliness": f"I am here, {username}. You are connected to me right now.",
        "Normal": f"Got it, {username}. How was your day otherwise?"
    }

    return {
        "mood": final_mood,
        "feedback": feedback_map.get(final_mood, f"I am listening, {username}."),
        "color_code": MOOD_COLORS[final_mood]
    }

def extract_themes(text):
    themes = []
    text = text.lower()
    if "work" in text or "job" in text: themes.append("Work")
    if "family" in text or "mom" in text or "dad" in text: themes.append("Family")
    if "sleep" in text or "tired" in text: themes.append("Health")
    if "love" in text or "relationship" in text: themes.append("Relationships")
    if not themes: themes.append("General Reflection")
    return themes