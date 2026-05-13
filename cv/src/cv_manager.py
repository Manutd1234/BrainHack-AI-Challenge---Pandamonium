"""Manages the CV model — Speed-Optimized Object Detection with YOLO11.

Uses YOLO11s (small) for maximum inference speed. Attempts TensorRT FP16
export for 3-5x additional speedup. Falls back to PyTorch if TensorRT
is unavailable.

Strategy: 75% accuracy / 25% speed scoring means a fast model that's
"good enough" beats a slow model that's perfect. yolo11s at 640px
runs ~6x faster than yolo11l at 1280px with only ~10% mAP drop.
"""

import io
import logging
import os
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# TIL-26 target classes (18 classes, index = category_id).
TIL26_CLASSES = {
    0: "cargo aircraft",
    1: "commercial aircraft",
    2: "drone",
    3: "fighter jet",
    4: "fighter plane",
    5: "helicopter",
    6: "light aircraft",
    7: "missile",
    8: "truck",
    9: "car",
    10: "tank",
    11: "bus",
    12: "van",
    13: "cargo ship",
    14: "yacht",
    15: "cruise ship",
    16: "warship",
    17: "sailboat",
}

# Inference image size — 640 is the YOLO sweet spot for speed.
# 1280 is 4x slower with diminishing accuracy returns.
INFER_IMGSZ = 640

# Map COCO classes to the closest TIL-26 classes.
# Required because the base YOLO model predicts COCO (80 classes),
# but the evaluator strictly expects TIL-26 (18 classes).
COCO_TO_TIL26_MAP = {
    2: 9,   # COCO car -> TIL26 car
    4: 1,   # COCO airplane -> TIL26 commercial aircraft
    5: 11,  # COCO bus -> TIL26 bus
    7: 8,   # COCO truck -> TIL26 truck
    8: 13,  # COCO boat -> TIL26 cargo ship
}

def _try_tensorrt_export(model_path: str, imgsz: int = INFER_IMGSZ) -> str | None:
    """Attempt to export a YOLO model to TensorRT format.

    TensorRT gives 3-5x inference speedup on NVIDIA GPUs.
    Returns the path to the exported engine, or None on failure.
    """
    engine_path = model_path.replace(".pt", ".engine")
    if os.path.exists(engine_path):
        logger.info(f"TensorRT engine already exists: {engine_path}")
        return engine_path

    try:
        model = YOLO(model_path)
        model.export(
            format="engine",
            imgsz=imgsz,
            half=True,       # FP16 for speed
            device=0,
        )
        if os.path.exists(engine_path):
            logger.info(f"TensorRT export successful: {engine_path}")
            return engine_path
    except Exception as e:
        logger.warning(f"TensorRT export failed ({e}), using PyTorch.")

    return None


class CVManager:

    def __init__(self):
        logger.info("Loading YOLO model...")

        # Priority order for model loading:
        # 1. Fine-tuned best.pt (trained on TIL-26 data) — highest priority
        # 2. Pre-downloaded yolo11s.pt (fast, good accuracy)
        # 3. Any available model as fallback
        model_candidates = [
            "best.pt",
            "models/best.pt",
            "yolo11s.pt",
            "yolo11n.pt",
        ]

        loaded_path = None
        for model_path in model_candidates:
            if os.path.exists(model_path):
                loaded_path = model_path
                break

        if loaded_path is None:
            # Download default model
            loaded_path = "yolo11s.pt"
            YOLO(loaded_path)  # triggers download

        # Try TensorRT export for speed (3-5x faster)
        engine_path = _try_tensorrt_export(loaded_path, imgsz=INFER_IMGSZ)
        if engine_path:
            self.model = YOLO(engine_path)
            logger.info(f"Using TensorRT engine: {engine_path}")
        else:
            self.model = YOLO(loaded_path)
            logger.info(f"Using PyTorch model: {loaded_path}")

        # Warm up the model with a dummy inference (eliminates first-call latency)
        try:
            dummy = np.zeros((INFER_IMGSZ, INFER_IMGSZ, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, imgsz=INFER_IMGSZ)
            logger.info("Model warm-up complete.")
        except Exception:
            pass

        # Check if the model has TIL-26 classes
        num_classes = len(getattr(self.model, "names", {}))
        self.is_finetuned = num_classes == 18
        logger.info(
            f"Model has {num_classes} classes. "
            f"Fine-tuned for TIL-26: {self.is_finetuned}"
        )

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes (JPEG format).

        Returns:
            A list of `dict`s, each with keys:
              - "bbox": [l, t, w, h] (LTWH format: left, top, width, height)
              - "category_id": int (class index matching TIL-26 target list)
        """

        img = Image.open(io.BytesIO(image))

        results = self.model.predict(
            img,
            verbose=False,
            imgsz=INFER_IMGSZ,    # 640 for speed (vs 1280)
            conf=0.15,            # Lower threshold to maximize recall for target classes
            iou=0.50,             # Raised IoU to better handle packed objects/convoys
            max_det=50,           # Cap detections for speed
            augment=False,        # No TTA — pure speed
            half=True,            # FP16 inference
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            # Batch extract all boxes at once (faster than per-box loop)
            if len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                original_cls = int(cls[i])
                
                # If model is not finetuned, map COCO -> TIL-26
                if not self.is_finetuned:
                    if original_cls not in COCO_TO_TIL26_MAP:
                        continue  # Skip detections that don't map to a target
                    mapped_cls = COCO_TO_TIL26_MAP[original_cls]
                else:
                    mapped_cls = original_cls
                
                x1, y1, x2, y2 = xyxy[i]
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "category_id": mapped_cls,
                })

        return detections
