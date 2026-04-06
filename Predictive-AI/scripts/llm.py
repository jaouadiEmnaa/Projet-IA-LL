from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
import pandas as pd
import joblib
import os
import xml.etree.ElementTree as ET
import logging

# --- LOGGER ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FEATURES ORDER (Must match your RF training) ---
FEATURES_ORDER = ['feeder', 'hour', 'material_part', 'designator', 'xy_interaction', 'nozzle']

# --- LOAD INVENTORY XML ---
def load_xml_data(part_no: int):
    try:
        tree = ET.parse('../data/inventaire.xml')
        root = tree.getroot()
        for item in root.findall('MATERIAL_DETAILS'):
            if item.get('PARTNO') == str(part_no):
                return {
                    "stock": item.get('CURRENT_QUANTITY'),
                    "loc": f"{item.get('ZONE_NAME')} / {item.get('STORAGE_UNIT_NAME')}",
                    "reel": item.get('REEL_ID')
                }
    except Exception as e:
        logger.error(f"XML parsing error: {e}")
    return None

# --- LOAD RANDOM FOREST MODEL ---
model_path = os.path.join('../notebooks/models', 'maintenance_model.pkl')  # replace with your RF file
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)
    logger.info("✅ Random Forest model loaded successfully")
else:
    logger.critical(f"❌ Model file not found at {model_path}")

# --- REQUEST SCHEMA ---
class PredictionRequest(BaseModel):
    designator: int
    nozzle: int
    feeder: int
    material_part: int
    x: float
    y: float
    hour: int
    
# --- OLLAMA RAG FUNCTION ---
def ask_local_llm(prompt: str):
    url = "http://127.0.0.1:11434/api/generate"
    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data, timeout=10)
        return response.json().get('response', "⚠️ No response from LLM")
    except Exception as e:
        logger.warning(f"LLM request failed: {e}")
        return "⚠️ Assistant unavailable (Check that Ollama is running)."
    
# --- MAIN ROUTE ---
@app.post("/api/predict-and-explain")
async def predict_and_explain(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    try:
        # 1. Feature engineering
        xy_interaction = req.x * req.y

        # 2. Prepare input dict
        input_data = {
            'feeder': req.feeder,
            'hour': req.hour,
            'material_part': req.material_part,
            'designator': req.designator,
            'xy_interaction': xy_interaction,
            'nozzle': req.nozzle
        }

        # 3. Convert to DataFrame
        features_df = pd.DataFrame([input_data])[FEATURES_ORDER]

        # 4. Prediction
        prob_failure = float(model.predict_proba(features_df)[0][1])
        status = "Fail" if prob_failure > 0.4 else "Pass"  # threshold can be adjusted

        # 5. Inventory info
        inventory_info = load_xml_data(req.material_part)
        inv_text = "No stock info found."
        if inventory_info:
            inv_text = f"Stock: {inventory_info['stock']}, Loc: {inventory_info['loc']}, Reel: {inventory_info['reel']}"

        # 6. LLM explanation
        explanation = "Machine operating within Pass parameters."
        if status == "Fail":
            prompt = f"""
            Role: Industrial Maintenance Expert.
            Failure Risk: {prob_failure*100:.1f}%. Status: {status}.
            Data: Position({req.x}, {req.y}), Feeder: {req.feeder}, Nozzle: {req.nozzle}.
            Inventory Context: {inv_text}.
            Action: Write a 2-line maximum technical instruction for the technician in English.
            """
            explanation = ask_local_llm(prompt)  # keep your existing LLM function

        return {
            "probability": round(prob_failure, 4),
            "status": status,
            "explanation": explanation,
            "inventory": inventory_info
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))