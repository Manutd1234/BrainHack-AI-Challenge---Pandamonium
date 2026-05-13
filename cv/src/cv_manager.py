"""Manages the CV model — Object Detection with YOLO11 + TensorRT.

Uses YOLO11l fine-tuned on TIL-26's 18 custom classes. Attempts
TensorRT export for 3-5x inference speedup on NVIDIA GPUs. Falls
back to native PyTorch if TensorRT is not available.
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


def _try_tensorrt_export(model_path: str, imgsz: int = 1280) -> str | None:
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
        # 1. Fine-tuned best.pt (trained on TIL-26 data)
        # 2. Pre-downloaded yolo11l.pt base model
        # 3. yolo11n.pt fallback
        model_candidates = [
            ("best.pt", True),
            ("models/best.pt", True),
            ("yolo11l.pt", False),
            ("yolo11n.pt", False),
        ]

        loaded_path = None
        for model_path, is_finetuned_candidate in model_candidates:
            if os.path.exists(model_path):
                loaded_path = model_path
                break

        if loaded_path is None:
            # Download default model
            loaded_path = "yolo11l.pt"
            YOLO(loaded_path)  # triggers download

        # Try TensorRT export for speed (3-5x faster)
        engine_path = _try_tensorrt_export(loaded_path, imgsz=1280)
        if engine_path:
            self.model = YOLO(engine_path)
            logger.info(f"Using TensorRT engine: {engine_path}")
        else:
            self.model = YOLO(loaded_path)
            logger.info(f"Using PyTorch model: {loaded_path}")

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

        try:
            results = self.model.predict(
                img,
                verbose=False,
                imgsz=1280,       # Higher resolution for small targets
                conf=0.15,        # Low confidence threshold for better recall
                iou=0.5,          # NMS IoU threshold
                max_det=100,      # Max detections per image
                augment=False,    # TTA disabled for speed
            )
        except Exception as e:
            logger.error(f"Inference failed ({e}), retrying on CPU.")
            self.model = YOLO("yolo11l.pt")
            results = self.model.predict(
                img,
                verbose=False,
                device="cpu",
                imgsz=1280,
                conf=0.15,
                iou=0.5,
            )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # Convert from xyxy (x1,y1,x2,y2) to LTWH (left,top,width,height)
                # as required by the TIL-26 spec.
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                l = x1
                t = y1
                w = x2 - x1
                h = y2 - y1
                category_id = int(box.cls[0].item())

                detections.append({
                    "bbox": [l, t, w, h],
                    "category_id": category_id,
                })

        return detections
