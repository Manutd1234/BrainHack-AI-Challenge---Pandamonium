import base64
import io
import logging
import os
import tempfile
from typing import Any, List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cv", "best.pt")
# Fallback to base model if best.pt doesn't exist during initial test
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolo11n.pt"

logger.info(f"Loading YOLO model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

app = FastAPI()

class CVRequest(BaseModel):
    instances: List[Dict[str, Any]]

def _predict_one(b64_image: str) -> List[Any]:
    if not model:
        return []
        
    raw = base64.b64decode(b64_image)
    image = Image.open(io.BytesIO(raw))
    
    results = model(image)
    
    # Extract bounding boxes
    boxes_out = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # xywh, conf, cls
            b = box.xywh[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            boxes_out.append([b[0], b[1], b[2], b[3], conf, cls])
            
    return boxes_out

@app.post("/cv")
def cv_endpoint(request: CVRequest):
    predictions = []
    for instance in request.instances:
        try:
            boxes = _predict_one(instance["b64"])
            predictions.append({"boxes": boxes})
        except Exception as e:
            logger.error(f"Error predicting instance: {e}")
            predictions.append({"error": str(e)})
            
    return {"predictions": predictions}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
