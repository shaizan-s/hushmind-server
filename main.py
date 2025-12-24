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
            # 🛡️ BULLETPROOF MODEL SELECTION
            # Try 1.5 Flash first (Best). If server is old, fall back to Pro.
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                print("✨ Using Gemini 1.5 Flash")
            except Exception:
                print("⚠️ 1.5 Flash failed, falling back to Gemini Pro")
                model = genai.GenerativeModel('gemini-pro')

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
    print("\n🚀 HUSHMIND SERVER IS LIVE (Bulletproof Mode) 🚀\n")
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
    
    try:
        ml_label = classifier.predict([request.text])[0]
        result = logic_core.resolve_mood(request.text, ml_label, request.username)
        return {
            "mood": result["mood"],
            "feedback": result["feedback"],
            "color_code": result["color_code"]
        }
    except Exception as e:
        print(f"Predict Error: {e}")
        return {"mood": "Neutral", "feedback": "I am listening.", "color_code": "0xFFFFFFFF"}

@app.post("/analyze_journal")
async def analyze_journal(request: TextRequest):
    # Safety Check
    is_crisis, crisis_msg = logic_core.check_safety(request.text)
    if is_crisis:
        return {"mood": "Crisis", "analysis": {"title": "Safety First", "summary": "Crisis detected.", "advice": "Please seek professional help.", "keywords": ["Help"]}}

    # Smart Mood
    raw_label = "Neutral"
    if classifier:
        raw_label = classifier.predict([request.text])[0]
    resolved_result = logic_core.resolve_mood(request.text, raw_label, request.username)
    smart_mood = resolved_result["mood"] 
        
    try:
        # 🛡️ BULLETPROOF SELECTION FOR JOURNAL TOO
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            model = genai.GenerativeModel('gemini-pro')

        prompt = f"""
        Analyze this diary entry: "{request.text}"
        Detected mood: {smart_mood}.
        Return ONLY JSON:
        {{
            "mood": "{smart_mood}", 
            "title": "Short poetic title",
            "summary": "One sentence summary",
            "advice": "One actionable step",
            "keywords": ["Tag1", "Tag2"]
        }}
        """
        response = model.generate_content(prompt)
        text_resp = response.text.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(text_resp)
        return {"mood": analysis.get("mood", smart_mood), "analysis": analysis}

    except Exception as e:
        print(f"AI Journal Error: {e}")
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

        # Send to Gemini
        chat = chat_manager.get_chat(user_id)
        response = chat.send_message(request.text)
        
        return {"reply": response.text, "is_crisis": False}

    except Exception as e:
        print(f"⚠️ Gemini Chat Error: {e}")
        return {
            "reply": "I'm having trouble connecting to my brain right now. Please try again in 1 minute.", 
            "is_crisis": False
        }