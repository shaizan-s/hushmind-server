from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
from groq import Groq # 🚀 NEW: Using Groq (Llama 3)
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import json

# Import our logic brain
import logic_core

# 1. Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not found. Please add it to Render Environment.")

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
        # Keep memory short to save space (last 10 messages)
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
    print("\n🚀 HUSHMIND SERVER IS LIVE (Powered by Llama 3) 🚀\n")
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
    # 1. Safety Check
    is_crisis, crisis_msg = logic_core.check_safety(request.text)
    if is_crisis:
        return {
            "mood": "Crisis",
            "analysis": {
                "title": "Safety First",
                "summary": "We detected distress.",
                "advice": "Please reach out to a professional immediately.",
                "keywords": ["Help", "Support"]
            }
        }

    # 2. Smart Mood Logic
    raw_label = "Neutral"
    if classifier:
        raw_label = classifier.predict([request.text])[0]
    resolved_result = logic_core.resolve_mood(request.text, raw_label, request.username)
    smart_mood = resolved_result["mood"]

    # 3. Llama 3 Analysis
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192", # ⚡ Fast & Smart Model
            messages=[
                {
                    "role": "system",
                    "content": "You are a psychologist. Analyze the diary entry. Return ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": f"""
                    Analyze this entry: "{request.text}"
                    The detected mood is: {smart_mood}.
                    
                    Return ONLY a JSON object with this format:
                    {{
                        "mood": "{smart_mood}", 
                        "title": "Short poetic title (3-5 words)",
                        "summary": "One warm sentence summary",
                        "advice": "One small positive action",
                        "keywords": ["Theme1", "Theme2"]
                    }}
                    """
                }
            ],
            temperature=0.5,
            response_format={"type": "json_object"} # Forces JSON
        )
        
        analysis = json.loads(completion.choices[0].message.content)
        return {"mood": analysis.get("mood", smart_mood), "analysis": analysis}

    except Exception as e:
        print(f"Llama Error: {e}")
        return {
            "mood": smart_mood,
            "analysis": {
                "title": "Daily Entry",
                "summary": "Saved successfully.",
                "advice": "Keep taking care of yourself.",
                "keywords": ["Journal"]
            }
        }

@app.post("/chat")
async def chat_with_ai(request: TextRequest):
    try:
        user_id = request.username
        
        # Safety Check
        is_crisis, crisis_msg = logic_core.check_safety(request.text)
        if is_crisis:
            return {"reply": crisis_msg, "mood": "Crisis", "is_crisis": True}

        # Update History
        chat_manager.add_message(user_id, "user", request.text)
        history = chat_manager.get_history(user_id)

        # Get Reply from Llama 3
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=history,
            temperature=0.7,
            max_tokens=150
        )
        
        ai_reply = completion.choices[0].message.content
        chat_manager.add_message(user_id, "assistant", ai_reply)
        
        return {"reply": ai_reply, "is_crisis": False}

    except Exception as e:
        print(f"Chat Error: {e}")
        return {
            "reply": "I'm having a little trouble thinking right now. Please try again.", 
            "is_crisis": False
        }