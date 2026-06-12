from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = "/workspace/model/best.pt"
IMGSZ = int(os.getenv("CV_IMGSZ", "1280"))
CONF = float(os.getenv("CV_CONF", "0.70"))
IOU = float(os.getenv("CV_IOU", "0.45"))
MAX_DET = int(os.getenv("CV_MAX_DET", "100"))


class CVManager:
    def __init__(self) -> None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device == "cuda"

        self.model = YOLO(MODEL_PATH)
        self.model.to(self.device)

    def predict_b64_batch(self, instances: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        images: list[np.ndarray | None] = []
        valid: list[bool] = []

        for inst in instances:
            b64 = inst.get("b64", "")
            try:
                img_bytes = base64.b64decode(b64)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    images.append(None)
                    valid.append(False)
                else:
                    images.append(img)
                    valid.append(True)
            except Exception:
                images.append(None)
                valid.append(False)

        preds: list[list[dict[str, Any]]] = [[] for _ in instances]
        good_idx = [i for i, ok in enumerate(valid) if ok]
        good_images = [images[i] for i in good_idx]

        if good_images:
            results = self.model.predict(
                good_images,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                max_det=MAX_DET,
                device=self.device,
                half=self.half,
                verbose=False,
                augment=False,
            )

            for slot, result, image in zip(good_idx, results, good_images):
                preds[slot] = self._format_result(result, image)

        return preds

    def _format_result(self, result, image: np.ndarray) -> list[dict[str, Any]]:
        if result.boxes is None or len(result.boxes) == 0:
            return []

        h, w = image.shape[:2]
        xywh = result.boxes.xywh.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)

        out: list[dict[str, Any]] = []
        for box, c in zip(xywh, cls):
            cx, cy, bw, bh = [float(v) for v in box]
            x = max(0.0, min(cx - bw / 2.0, w - 1.0))
            y = max(0.0, min(cy - bh / 2.0, h - 1.0))
            bw = max(0.0, min(bw, w - x))
            bh = max(0.0, min(bh, h - y))
            if bw <= 0.0 or bh <= 0.0:
                continue
            out.append(
                {
                    "bbox": [x, y, bw, bh],
                    "category_id": int(c),
                }
            )
        return out
