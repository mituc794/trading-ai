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
    title="Crypto Brain API (GPT-OSS Edition)",
    description="Sentiment (GPT-OSS 20B) + Reasoning (GPT-OSS 120B)",
    version="3.0.1"
)

# ==========================================
# 0. SECURITY & CONFIGURATION
# ==========================================
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

class LastPrediction(BaseModel):
    action: str
    confidence: float
    reason: Optional[str] = None
    created_at: Optional[str] = None 

class PredictRequest(BaseModel):
    symbol: str
    current_price: float
    last_50_candles: List[Candle]
    sentiment_context: Optional[str] = "Neutral"
    last_prediction: Optional[LastPrediction] = None

class PredictResponse(BaseModel):
    action: str
    confidence: float
    reason: str
    trend: str

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    system_instruction: Optional[str] = "You are a helpful AI assistant."

class SummaryRequest(BaseModel):
    text: str
    target_length: Optional[str] = "one short sentence"

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def format_market_data(candles: List[Candle]) -> str:
    closes = [str(round(c.close, 2)) for c in candles[-30:]] 
    return f"- Closing Prices: {', '.join(closes)}"

def clean_json_response(text: str):
    """
    Fallback cleaner in case Groq sends back raw text despite JSON mode.
    """
    # Remove <think> blocks common in reasoning models
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "")
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except:
            pass
    return {"action": "HOLD", "confidence": 0.0, "reason": "JSON Parse Error", "trend": "Unknown"}

# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {"status": "Online 🚀", "provider": "Groq (GPT-OSS)"}

# ✅ RESTORED HEALTH ENDPOINT
@app.get("/health")
def health_check():
    return {"status": "online"}

# --- SENTIMENT (GPT-OSS 20B) ---
@app.post("/sentiment", dependencies=[Security(get_api_key)])
def analyze_sentiment(req: SentimentRequest):
    if not groq_client: raise HTTPException(500, "Groq API Key not set")

    safe_text = req.text[:4000]

    system_prompt = """
    You are a financial sentiment analyzer. 
    You MUST respond with valid JSON only. 
    Do not output any thinking tags or conversational text.
    """

    user_prompt = f"""
    Analyze the sentiment of this text: "{safe_text}"
    
    Return JSON format:
    {{
        "label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
        "score": 0.0 to 1.0
    }}
    """

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_completion_tokens=500,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Sentiment Error: {e}")
        return {"label": "NEUTRAL", "score": 0.0}

# --- PREDICT (GPT-OSS 120B) ---
@app.post("/predict", response_model=PredictResponse, dependencies=[Security(get_api_key)])
def predict_market(req: PredictRequest):
    if not groq_client: raise HTTPException(500, "Groq API Key not set")

    context_str = "No previous trade."
    if req.last_prediction:
        context_str = f"""
        PREVIOUS TRADE: {req.last_prediction.action} 
        CONFIDENCE: {req.last_prediction.confidence}
        REASON: {req.last_prediction.reason}
        """

    system_prompt = """
    You are an expert Crypto Trader.
    Output a SINGLE valid JSON object.
    Do not include markdown formatting or explanations outside the JSON.
    """

    user_prompt = f"""
    DATA:
    Symbol: {req.symbol}
    Price: {req.current_price}
    Sentiment: {req.sentiment_context}
    Candles: {format_market_data(req.last_50_candles)}
    
    {context_str}

    REQUIRED JSON OUTPUT:
    {{
        "trend": "Bullish" or "Bearish",
        "action": "BUY" or "SELL" or "HOLD",
        "confidence": 0.0 to 1.0,
        "reason": "Short explanation"
    }}
    """

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            reasoning_effort="medium",
            response_format={"type": "json_object"}
        )
        
        content = completion.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(500, f"Groq Error: {str(e)}")

# --- CHAT (GPT-OSS 120B) ---
@app.post("/chat", dependencies=[Security(get_api_key)])
def chat_generate(req: ChatRequest):
    if not groq_client: raise HTTPException(500, "Groq API Key not set")

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": req.system_instruction},
                {"role": "user", "content": req.prompt}
            ],
            temperature=req.temperature,
            max_completion_tokens=req.max_tokens,
            reasoning_effort="medium" 
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(500, f"Groq Error: {str(e)}")

# --- SUMMARY (GPT-OSS 20B) ---
@app.post("/summary", dependencies=[Security(get_api_key)])
def generate_summary(req: SummaryRequest):
    if not groq_client: raise HTTPException(500, "Groq API Key not set")

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "Summarize concisely."},
                {"role": "user", "content": f"Summarize this text into {req.target_length}:\n\n{req.text}"}
            ],
            temperature=0.5,
            max_completion_tokens=1024
        )
        return {"summary": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(500, f"Groq Error: {str(e)}")
