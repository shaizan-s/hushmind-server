from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import google.generativeai as genai
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import json

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
            # ✅ CORRECTED MODEL NAME: gemini-1.5-flash
            model = genai.GenerativeModel('gemini-1.5-flash')
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
    print("\n🚀 HUSHMIND SERVER IS RUNNING (Dual-Brain Mode) 🚀\n")
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

# ⚡ FAST BRAIN: For Mood Graph & Chat Colors
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

# 🧠 DEEP BRAIN: For Diary Analysis (Title, Advice, Summary)
@app.post("/analyze_journal")
async def analyze_journal(request: TextRequest):
    # 1. Safety Check First
    is_crisis, crisis_msg = logic_core.check_safety(request.text)
    if is_crisis:
        return {
            "mood": "Crisis",
            "analysis": {
                "title": "Safety First",
                "summary": "We detected distress in your message.",
                "advice": "Please reach out to a professional or helpline immediately.",
                "keywords": ["Help", "Support", "Safety"]
            }
        }

    # 2. Get Basic Mood (Backup)
    ml_label = "Neutral"
    if classifier:
        ml_label = classifier.predict([request.text])[0]
        
    # 3. Ask Gemini for "Therapist Report"
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are a warm, empathetic psychologist. Analyze this diary entry: "{request.text}"
        
        Return ONLY a JSON object (no markdown) with this exact format:
        {{
            "mood": "One word mood (e.g. Happy, Anxious, Calm, Proud)",
            "title": "A short, poetic title for this entry (3-5 words)",
            "summary": "A warm, 1-sentence validation of how they feel.",
            "advice": "One small, actionable, positive step they can take right now.",
            "keywords": ["Theme1", "Theme2", "Theme3"]
        }}
        """
        response = model.generate_content(prompt)
        text_resp = response.text.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(text_resp)
        
        return {
            "mood": analysis.get("mood", ml_label),
            "analysis": analysis 
        }

    except Exception as e:
        print(f"AI Error: {e}")
        return {
            "mood": ml_label,
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