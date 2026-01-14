from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import requests # For calling Hugging Face API
from groq import Groq 
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import json

# Import our logic brain
import logic_core

# 1. Load Environment Variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") 

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not found.")
if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN not found. AI Safety check will be skipped.")

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

# Hugging Face API Setup
HF_API_URL = "https://api-inference.huggingface.co/models/sisyphus/bert-base-uncased-suicidality"
hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}

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

# ✨ Helper function to check Crisis via Cloud AI
def check_crisis_with_ai(text):
    if not HF_TOKEN: return False 
    
    try:
        response = requests.post(HF_API_URL, headers=hf_headers, json={"inputs": text})
        api_output = response.json()
        
        if isinstance(api_output, list) and len(api_output) > 0:
            top_result = api_output[0][0] 
            label = top_result['label']
            score = top_result['score']
            
            # 🚨 CHANGED: Threshold lowered to 0.4 (40%) to catch subtle phrases
            if label == "Suicide" and score > 0.4: 
                print(f"🚨 AI DETECTED CRISIS: {label} ({score})")
                return True
    except Exception as e:
        print(f"⚠️ AI Safety Check Failed: {e}")
    
    return False

# 4. Server Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 HUSHMIND SERVER IS LIVE (Hybrid AI) 🚀\n")
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
    # 1. Check AI Safety First
    if check_crisis_with_ai(request.text):
         return {
            "mood": "Crisis",
            "feedback": "I'm detecting that you are in severe distress. Please reach out to the safety contacts.",
            "color_code": "#FF5252" # Red
         }

    # 2. Normal Logic
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
    # 1. Basic Keyword Check
    is_crisis_keyword, _ = logic_core.check_safety(request.text)
    
    # 2. Advanced AI Check
    is_crisis_ai = check_crisis_with_ai(request.text)

    if is_crisis_keyword or is_crisis_ai:
        return {"mood": "Crisis", "analysis": {"title": "Safety First", "summary": "Crisis detected.", "advice": "Please seek professional help immediately.", "keywords": ["Help", "Safety"]}}

    # 3. Normal Logic
    raw_label = "Neutral"
    if classifier:
        raw_label = classifier.predict([request.text])[0]
    resolved_result = logic_core.resolve_mood(request.text, raw_label, request.username)
    smart_mood = resolved_result["mood"]

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
        
        # 1. Basic Keyword Check
        is_crisis_keyword, crisis_msg = logic_core.check_safety(request.text)
        if is_crisis_keyword:
             return {"reply": crisis_msg, "mood": "Crisis", "is_crisis": True}

        # 2. Advanced AI Check
        if check_crisis_with_ai(request.text):
            return {
                "reply": "It sounds like you are going through a critical time. I want to make sure you are safe. Please connect with your emergency contact.", 
                "mood": "Crisis", 
                "is_crisis": True
            }

        # 3. Normal Chat Logic
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