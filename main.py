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
# ⚠️ If this still fails on Render, you can hardcode the key here temporarily:
# GROQ_API_KEY = "gsk_..." 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not found.")

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

# 2. Chat Memory Management
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

# 3. Load ML Model
try:
    with open("mood_model.pkl", "rb") as f:
        classifier = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    classifier = None

# 4. Server Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 HUSHMIND SERVER IS LIVE (Llama 3.3 Fixed) 🚀\n")
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
    if not classifier:
        return {"error": "Model not loaded"}
        
    ml_label = classifier.predict([request.text])[0]
    result = logic_core.resolve_mood(request.text, ml_label, request.username)
    return {
        "mood": result["mood"],
        "feedback": result["feedback"],
        "color_code": result["color_code"]
    }

@app.post("/analyze_journal")
async def analyze_journal(request: TextRequest):
    is_crisis, crisis_msg = logic_core.check_safety(request.text)
    if is_crisis:
        return {"mood": "Crisis", "analysis": {"title": "Safety First", "summary": "Crisis detected.", "advice": "Seek help.", "keywords": ["Help"]}}

    raw_label = "Neutral"
    if classifier:
        raw_label = classifier.predict([request.text])[0]
    resolved_result = logic_core.resolve_mood(request.text, raw_label, request.username)
    smart_mood = resolved_result["mood"]

    try:
        completion = client.chat.completions.create(
            # ✅ UPDATED MODEL NAME
            model="llama-3.3-70b-versatile", 
            messages=[
                {
                    "role": "system",
                    "content": "You are a psychologist. Return ONLY valid JSON."
                },
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
        is_crisis, crisis_msg = logic_core.check_safety(request.text)
        if is_crisis:
            return {"reply": crisis_msg, "mood": "Crisis", "is_crisis": True}

        chat_manager.add_message(user_id, "user", request.text)
        history = chat_manager.get_history(user_id)

        completion = client.chat.completions.create(
            # ✅ UPDATED MODEL NAME
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