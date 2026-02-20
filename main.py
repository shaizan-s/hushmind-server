from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import requests 
from groq import Groq 
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import json
import logic_core

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") 

client = Groq(api_key=GROQ_API_KEY)
HF_API_URL = "https://api-inference.huggingface.co/models/sisyphus/bert-base-uncased-suicidality"
hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

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

try:
    with open("mood_model.pkl", "rb") as f:
        classifier = pickle.load(f)
except Exception:
    classifier = None

def check_crisis_with_ai(text):
    if not HF_TOKEN: return False 
    try:
        response = requests.post(HF_API_URL, headers=hf_headers, json={"inputs": text})
        api_output = response.json()
        if isinstance(api_output, list) and len(api_output) > 0:
            top_result = api_output[0][0] 
            if top_result['label'] == "Suicide" and top_result['score'] > 0.4: 
                return True
    except Exception:
        pass
    return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class TextRequest(BaseModel):
    text: str
    username: str = "Friend"

@app.post("/predict")
def predict_mood(request: TextRequest):
    if check_crisis_with_ai(request.text):
         return {"mood": "Crisis", "feedback": "I'm detecting severe distress. Please reach out to safety contacts.", "color_code": "#FF5252"}
    ml_label = classifier.predict([request.text])[0] if classifier else "Normal"
    return logic_core.resolve_mood(request.text, ml_label, request.username)

@app.post("/analyze_journal")
async def analyze_journal(request: TextRequest):
    is_crisis_keyword, _ = logic_core.check_safety(request.text)
    if is_crisis_keyword or check_crisis_with_ai(request.text):
        return {"mood": "Crisis", "analysis": {"title": "Safety First", "summary": "Crisis detected.", "advice": "Please seek help.", "keywords": ["Safety"]}}
    
    raw_label = classifier.predict([request.text])[0] if classifier else "Normal"
    smart_mood = logic_core.resolve_mood(request.text, raw_label, request.username)["mood"]

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "system", "content": "You are a psychologist. Return ONLY valid JSON."},
                      {"role": "user", "content": f'Analyze: "{request.text}" Mood: {smart_mood}. Return JSON: {{ "mood": "{smart_mood}", "title": "...", "summary": "...", "advice": "...", "keywords": ["..."] }}'}],
            temperature=0.5, response_format={"type": "json_object"}
        )
        return {"mood": smart_mood, "analysis": json.loads(completion.choices[0].message.content)}
    except:
        return {"mood": smart_mood, "analysis": {"title": "Daily Entry", "summary": "Saved.", "advice": "Keep going.", "keywords": ["Journal"]}}

@app.post("/chat")
async def chat_with_ai(request: TextRequest):
    is_crisis_keyword, crisis_msg = logic_core.check_safety(request.text)
    if is_crisis_keyword or check_crisis_with_ai(request.text):
        return {"reply": crisis_msg if is_crisis_keyword else "Please connect with your emergency contact.", "mood": "Crisis", "is_crisis": True}

    chat_manager.add_message(request.username, "user", request.text)
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=chat_manager.get_history(request.username), temperature=0.7, max_tokens=150)
        ai_reply = completion.choices[0].message.content
        chat_manager.add_message(request.username, "assistant", ai_reply)
        return {"reply": ai_reply, "is_crisis": False}
    except:
        return {"reply": "I'm having trouble thinking right now.", "is_crisis": False}