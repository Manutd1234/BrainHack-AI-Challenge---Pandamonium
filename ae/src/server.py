import logging
import os
import numpy as np
from typing import Any, List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ae", "policy.zip")
logger.info(f"Loading PPO model from {MODEL_PATH}...")
try:
    model = PPO.load(MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load AE model: {e}")
    model = None

app = FastAPI()

class AERequest(BaseModel):
    state: List[float] # Matches the environment observation space shape

@app.post("/ae")
def ae_endpoint(request: AERequest):
    if not model:
        return {"action": 0} # Dummy fallback
    
    # Predict action from state
    # stable-baselines3 expects a numpy array
    obs = np.array(request.state)
    action, _states = model.predict(obs, deterministic=True)
    return {"action": int(action)}

@app.get("/reset")
def reset():
    # Placeholder for environment reset logic if the agent maintains state
    return {"status": "reset"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
