import os
import json
import re
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from groq import Groq

# Initialize FastAPI
app = FastAPI(
    title="Crypto Brain API (Groq Edition)",
    description="Sentiment (Llama 8B) + Reasoning (DeepSeek 70B)",
    version="2.0.0"
)

# ==========================================
# 0. SECURITY & CONFIGURATION
# ==========================================
# Get these from Render Environment Variables
API_KEY = os.getenv("API_KEY", "default-insecure-key")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="⛔ Invalid or Missing API Key")

# ==========================================
# 1. LOAD CLIENTS
# ==========================================
print("⏳ System Startup: Connecting to Groq...")

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq Client: Connected")
else:
    groq_client = None
    print("❌ Groq Client: Disabled (Missing GROQ_API_KEY)")

# ==========================================
# 2. DATA MODELS
# ==========================================
class SentimentRequest(BaseModel):
    text: str

class Candle(BaseModel):
    close: float
    open: float
    high: float
    low: float
    volume: float

class PredictRequest(BaseModel):
    symbol: str
    current_price: float
    last_50_candles: List[Candle]
    sentiment_context: Optional[str] = "Neutral"

class PredictResponse(BaseModel):
    action: str
    confidence: float
    reason: str
    trend: str

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.6
    system_instruction: Optional[str] = "You are a helpful AI assistant."

class SummaryRequest(BaseModel):
    text: str
    target_length: Optional[str] = "one short sentence"

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def format_market_data(candles: List[Candle]) -> str:
    # Reduce data size to save tokens context
    closes = [str(round(c.close, 2)) for c in candles[-30:]] # Last 30 candles
    return f"- Closing Prices: {', '.join(closes)}"

def clean_json_response(text: str):
    """
    Cleans the DeepSeek/Llama thinking tokens and extracts valid JSON.
    """
    # Remove DeepSeek <think> blocks if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Try to find JSON inside code blocks
    match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if match: return json.loads(match.group(1))
    
    # Try to find raw JSON brackets
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match: return json.loads(match.group(0))
    
    # Fallback
    return {"error": "Failed to parse JSON", "raw": text}

# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {"status": "Online 🚀", "provider": "Groq"}

@app.get("/health")
def health_check():
    return {"status": "online"}

# --- SENTIMENT (Llama 3 8B) ---
@app.post("/sentiment", dependencies=[Security(get_api_key)])
def analyze_sentiment(req: SentimentRequest):
    if not groq_client: raise HTTPException(500, "Groq API Key not set")

    # Prompt designed to mimic the original Classifier output (Label + Score)
    # but with the added intelligence of an LLM.
    prompt = f"""
    Analyze the sentiment of this crypto text: "{req.text}"
    
    Return JSON ONLY with these exact fields:
    - label: "POSITIVE" or "NEGATIVE" or "NEUTRAL"
    - score: A float between 0.0 and 1.0 representing intensity.
    """

    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Fast model for simple tasks
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(500, f"Groq Error: {str(e)}")

# --- PREDICT (DeepSeek 70B) ---
@app.post("/predict", response_model=PredictResponse, dependencies=[Security(get_api_key)])
def predict_market(req: PredictRequest):
    if not groq_client: raise HTTPException(500, "Groq API Key not set")

    prompt = f"""
    Act as a professional Crypto Trader. Output JSON ONLY.
    SYMBOL: {req.symbol} | SENTIMENT: {req.sentiment_context}
    DATA: {format_market_data(req.last_50_candles)}

    Response Format:
    {{ "trend": "Bullish/Bearish", "action": "BUY/SELL/HOLD", "confidence": 0.0-1.0, "reason": "Short reason" }}
    """

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b", # The "Reasoning" Model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6, # DeepSeek recommends 0.6 for reasoning
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        content = completion.choices[0].message.content
        return clean_json_response(content)
    except Exception as e:
        raise HTTPException(500, f"Groq Error: {str(e)}")

# --- CHAT (Llama 3 70B) ---
@app.post("/chat", dependencies=[Security(get_api_key)])
def chat_generate(req: ChatRequest):
    if not groq_client: raise HTTPException(500, "Groq API Key not set")

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Best general purpose chat model
            messages=[
                {"role": "system", "content": req.system_instruction},
                {"role": "user", "content": req.prompt}
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(500, f"Groq Error: {str(e)}")

# --- SUMMARY (Llama 3 8B) ---
@app.post("/summary", dependencies=[Security(get_api_key)])
def generate_summary(req: SummaryRequest):
    if not groq_client: raise HTTPException(500, "Groq API Key not set")

    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192", # Fast for summarization
            messages=[
                {"role": "system", "content": "You are a precise summarizer. Output ONLY the summary."},
                {"role": "user", "content": f"Summarize this text into {req.target_length}:\n\n{req.text}"}
            ],
            temperature=0.1,
            max_tokens=2048
        )
        return {"summary": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(500, f"Groq Error: {str(e)}")
