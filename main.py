from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from groq import Groq 
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import json
import logic_core

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

class ChatManager:
    def __init__(self):
        self.histories = {}
    def get_history(self, user_id):
        if user_id not in self.histories:
            self.histories[user_id] = [{"role": "system", "content": "You are HushMind, a warm AI friend. Keep answers short."}]
        return self.histories[user_id]
    def add_message(self, user_id, role, content):
        history = self.get_history(user_id)
        history.append({"role": role, "content": content})
        if len(history) > 11: 
            self.histories[user_id] = [history[0]] + history[-10:]

chat_manager = ChatManager()

# --- THE NEW SMART CLASSIFIER ---
def get_smart_mood(text):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a psychological classifier. Read the text and reply with EXACTLY ONE of these words and nothing else: Depression, Anxiety, Stress, Loneliness, Happy, Calm, Normal. Do not explain."
                },
                {"role": "user", "content": text}
            ],
            temperature=0.1, # Low temperature means strict, predictable output
            max_tokens=10
        )
        mood = completion.choices[0].message.content.strip()
        # Clean up any weird punctuation Llama might add
        for valid_mood in logic_core.MOOD_COLORS.keys():
            if valid_mood.lower() in mood.lower():
                return valid_mood
        return "Normal"
    except Exception as e:
        print(f"Groq Classification Error: {e}")
        return "Normal"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 HUSHMIND SERVER IS LIVE (GROQ SEMANTIC AI) 🚀\n")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class TextRequest(BaseModel):
    text: str
    username: str = "Friend"

@app.post("/predict")
def predict_mood(request: TextRequest):
    # 1. Check Hardcoded Crisis Keywords
    is_crisis, crisis_msg = logic_core.check_safety(request.text)
    if is_crisis:
         return logic_core.get_feedback_and_color("Crisis", request.username)
    
    # 2. Let Groq (Llama 3) determine the exact mood based on context
    smart_mood = get_smart_mood(request.text)
    
    # 3. Return the right colors and feedback
    return logic_core.get_feedback_and_color(smart_mood, request.username)

@app.post("/chat")
async def chat_with_ai(request: TextRequest):
    is_crisis, crisis_msg = logic_core.check_safety(request.text)
    if is_crisis:
        return {"reply": crisis_msg, "mood": "Crisis", "is_crisis": True}

    chat_manager.add_message(request.username, "user", request.text)
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=chat_manager.get_history(request.username), temperature=0.7, max_tokens=150)
        ai_reply = completion.choices[0].message.content
        chat_manager.add_message(request.username, "assistant", ai_reply)
        return {"reply": ai_reply, "is_crisis": False}
    except:
        return {"reply": "I'm having trouble thinking right now.", "is_crisis": False}
    
@app.post("/analyze_journal")
async def analyze_journal(data: dict):
    journal_text = data.get("text", "")
    username = data.get("username", "Friend")

    prompt = f"""
    You are a compassionate mental health companion for {username}.
    Analyze this journal entry: "{journal_text}"
    
    1. Classify mood: Calm, Anxious, Stressed, Depressed, Lonely, or Crisis.
    2. Provide a short, warm, empathetic insight (max 2 sentences).
    3. If there is self-harm risk, the mood MUST be 'Crisis'.

    Return ONLY a JSON object:
    {{
      "mood": "MoodLabel",
      "insight": "Your empathetic reflection here..."
    }}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Journal Error: {e}")
        return {"mood": "Neutral", "insight": "I'm here for you. Thank you for sharing."}