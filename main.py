from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import google.generativeai as genai
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Import our new "Brain"
import logic_core

# 1. Load Environment Variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY not found in .env file")
else:
    genai.configure(api_key=API_KEY)

# 2. Chat Memory Management
class ChatManager:
    def __init__(self):
        self.sessions = {}

    def get_chat(self, user_id):
        if user_id not in self.sessions:
            model = genai.GenerativeModel('gemini-2.5-flash')
            self.sessions[user_id] = model.start_chat(history=[
                {"role": "user", "parts": ["You are HushMind, a warm, empathetic mental health AI friend. Keep answers short (max 2-3 sentences)."]},
                {"role": "model", "parts": ["Understood. I am HushMind, here to listen and support."]}
            ])
        return self.sessions[user_id]

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
    print("\n🚀 HUSHMIND SERVER IS RUNNING (Refactored & Secure) 🚀\n")
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

@app.post("/chat")
async def chat_with_ai(request: TextRequest):
    try:
        user_id = request.username
        
        # 1. 🚨 SAFETY CHECK FIRST
        is_crisis, crisis_msg = logic_core.check_safety(request.text)
        if is_crisis:
            return {
                "reply": crisis_msg,
                "mood": "Crisis",
                "is_crisis": True
            }

        # 2. Send to Gemini
        chat = chat_manager.get_chat(user_id)
        response = chat.send_message(request.text)
        
        return {
            "reply": response.text, 
            "is_crisis": False
        }

    except Exception as e:
        print(f"⚠️ Gemini Error: {e}")
        return {
            "reply": "I'm feeling a bit foggy right now. Can we try again in a moment?", 
            "is_crisis": False
        }