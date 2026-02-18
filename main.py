from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
from groq import Groq 
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import json

# Import our logic brain
import logic_core

# 1. Load Environment Variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not found.")

# Initialize Groq Client (For Chatting)
client = Groq(api_key=GROQ_API_KEY)

# 2. LOAD THE TWO BRAINS 🧠🧠
try:
    with open("suicide_model.pkl", "rb") as f:
        suicide_model = pickle.load(f)
    print("✅ Guard Dog (Suicide Model) loaded successfully")
except Exception as e:
    print(f"❌ Error loading suicide_model.pkl: {e}")
    suicide_model = None

try:
    with open("mood_model.pkl", "rb") as f:
        mood_model = pickle.load(f)
    print("✅ Therapist (Mood Model) loaded successfully")
except Exception as e:
    print(f"❌ Error loading mood_model.pkl: {e}")
    mood_model = None


# 3. Chat Memory Management
class ChatManager:
    def __init__(self):
        self.histories = {}

    def get_history(self, user_id):
        if user_id not in self.histories:
            self.histories[user_id] = [
                {"role": "system", "content": "You are HushMind, a warm, empathetic mental health AI friend. Keep answers short (max 2-3 sentences). Be supportive."}
            ]
        return self.histories[user_id]

    def add_message(self, user_id, role, content):
        history = self.get_history(user_id)
        history.append({"role": role, "content": content})
        if len(history) > 11: 
            self.histories[user_id] = [history[0]] + history[-10:]

chat_manager = ChatManager()


# 4. Server Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 HUSHMIND SERVER IS LIVE (Dual-Brain Architecture) 🚀\n")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    username: str = "Friend"

# --- ENDPOINTS ---

@app.post("/predict")
def predict_mood(request: TextRequest):
    # ✨ MAGIC: Pass BOTH models to the logic core
    # The logic_core will ask the Guard Dog first, then the Therapist.
    result = logic_core.resolve_mood(request.text, suicide_model, mood_model, request.username)
    return result

@app.post("/analyze_journal")
async def analyze_journal(request: TextRequest):
    # 1. Get Smart Mood using Dual-Brain Logic
    resolved_result = logic_core.resolve_mood(request.text, suicide_model, mood_model, request.username)
    smart_mood = resolved_result["mood"]
    
    # 2. If Crisis, Stop Here
    if smart_mood == "Crisis":
        return {
            "mood": "Crisis", 
            "analysis": {
                "title": "Safety First", 
                "summary": "We detected distress in your entry.", 
                "advice": "Please reach out to your safety contact immediately.", 
                "keywords": ["Help", "Safety"]
            }
        }

    # 3. If Safe, Get Deep Analysis from Groq (Llama 3)
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": "You are a psychologist. Return ONLY valid JSON."},
                {
                    "role": "user", 
                    "content": f"""
                    Analyze this: "{request.text}"
                    Mood: {smart_mood}.
                    Return JSON: {{ "mood": "{smart_mood}", "title": "...", "summary": "...", "advice": "...", "keywords": ["..."] }}
                    """
                }
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(completion.choices[0].message.content)
        return {"mood": analysis.get("mood", smart_mood), "analysis": analysis}

    except Exception as e:
        print(f"Llama Error: {e}")
        return {"mood": smart_mood, "analysis": {"title": "Daily Entry", "summary": "Saved.", "advice": "Keep going.", "keywords": ["Journal"]}}

@app.post("/chat")
async def chat_with_ai(request: TextRequest):
    try:
        user_id = request.username
        
        # 1. Check Safety First (Dual-Brain)
        resolved = logic_core.resolve_mood(request.text, suicide_model, mood_model, user_id)
        
        if resolved["mood"] == "Crisis":
             return {
                 "reply": resolved["feedback"], 
                 "mood": "Crisis", 
                 "is_crisis": True
             }

        # 2. Normal Chat Logic (If Safe)
        chat_manager.add_message(user_id, "user", request.text)
        history = chat_manager.get_history(user_id)

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history,
            temperature=0.7,
            max_tokens=150
        )
        
        ai_reply = completion.choices[0].message.content
        chat_manager.add_message(user_id, "assistant", ai_reply)
        return {"reply": ai_reply, "is_crisis": False}

    except Exception as e:
        print(f"Chat Error: {e}")
        return {"reply": "I'm having trouble thinking right now.", "is_crisis": False}