"""YOLO26x + SAHI CV manager for TIL-AI 2026."""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)
MODEL_PATH = os.getenv("CV_MODEL_PATH", "/app/model/best.pt")
DEFAULT_CONF = float(os.getenv("CV_DEFAULT_CONF", "0.20"))
DEFAULT_IOU = float(os.getenv("CV_IOU", "0.45"))
SAHI_MIN_SIZE = int(os.getenv("CV_SAHI_MIN_SIZE", "640"))
SAHI_SLICE_SIZE = int(os.getenv("CV_SAHI_SLICE_SIZE", "640"))
SAHI_OVERLAP = float(os.getenv("CV_SAHI_OVERLAP", "0.20"))

CLASS_NAMES = [
    "cargo aircraft",
    "commercial aircraft",
    "drone",
    "fighter jet",
    "fighter plane",
    "helicopter",
    "light aircraft",
    "missile",
    "truck",
    "car",
    "tank",
    "bus",
    "van",
    "cargo ship",
    "yacht",
    "cruise ship",
    "warship",
    "sailboat",
]

CLASS_CONF = {
    0: 0.25,
    1: 0.25,
    2: 0.15,
    3: 0.25,
    4: 0.25,
    5: 0.20,
    6: 0.20,
    7: 0.15,
    8: 0.25,
    9: 0.25,
    10: 0.25,
    11: 0.25,
    12: 0.25,
    13: 0.25,
    14: 0.20,
    15: 0.25,
    16: 0.25,
    17: 0.20,
}


class CVManager:
    """Loads a YOLO26x checkpoint and returns TIL-format detections."""

    def __init__(self) -> None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"CV checkpoint not found at {MODEL_PATH}. Run cv_train.py and "
                "copy model/best.pt into cv/model/best.pt before building."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info("Loading YOLO checkpoint from %s on %s", MODEL_PATH, self.device)
        self.model = YOLO(MODEL_PATH)
        self.model.to(self.device)

        self.sahi_model = self._load_sahi_model()
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        LOGGER.info("CVManager ready")

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Perform object detection on one JPEG image."""
        frame = self._decode(image)
        height, width = frame.shape[:2]

        if self.sahi_model is not None and max(height, width) >= SAHI_MIN_SIZE:
            return self._predict_sahi(frame)

        return self._predict_full_image(frame)

    def _load_sahi_model(self):
        try:
            from sahi import AutoDetectionModel

            model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=MODEL_PATH,
                confidence_threshold=DEFAULT_CONF,
                device=self.device,
            )
            LOGGER.info("SAHI model wrapper loaded")
            return model
        except Exception as exc:
            LOGGER.warning("SAHI unavailable, falling back to full-image inference: %s", exc)
            return None

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes as JPEG")
        return image

    def _predict_full_image(self, image: np.ndarray) -> list[dict[str, Any]]:
        result = self.model.predict(
            image,
            conf=DEFAULT_CONF,
            iou=DEFAULT_IOU,
            verbose=False,
            device=self.device,
        )[0]
        return self._format_yolo_result(result, image.shape)

    def _predict_sahi(self, image: np.ndarray) -> list[dict[str, Any]]:
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            image,
            self.sahi_model,
            slice_height=SAHI_SLICE_SIZE,
            slice_width=SAHI_SLICE_SIZE,
            overlap_height_ratio=SAHI_OVERLAP,
            overlap_width_ratio=SAHI_OVERLAP,
            perform_standard_pred=True,
            postprocess_type="NMS",
            postprocess_match_threshold=DEFAULT_IOU,
            postprocess_class_agnostic=False,
            verbose=0,
        )

        predictions = []
        height, width = image.shape[:2]
        for obj in result.object_prediction_list:
            category_id = int(obj.category.id)
            if category_id < 0 or category_id >= len(CLASS_NAMES):
                continue
            if obj.score.value < CLASS_CONF.get(category_id, DEFAULT_CONF):
                continue

            left = int(round(max(0, min(obj.bbox.minx, width - 1))))
            top = int(round(max(0, min(obj.bbox.miny, height - 1))))
            right = int(round(max(0, min(obj.bbox.maxx, width))))
            bottom = int(round(max(0, min(obj.bbox.maxy, height))))
            box_width = right - left
            box_height = bottom - top
            if box_width <= 0 or box_height <= 0:
                continue

            predictions.append(
                {
                    "bbox": [left, top, box_width, box_height],
                    "category_id": category_id,
                }
            )

        return predictions

    def _format_yolo_result(
        self,
        result,
        image_shape: tuple[int, int, int],
    ) -> list[dict[str, Any]]:
        if result.boxes is None or len(result.boxes) == 0:
            return []

        height, width = image_shape[:2]
        boxes_xywh = result.boxes.xywh.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()

        predictions = []
        for xywh, category_id, confidence in zip(boxes_xywh, class_ids, confidences):
            if category_id < 0 or category_id >= len(CLASS_NAMES):
                continue
            if confidence < CLASS_CONF.get(int(category_id), DEFAULT_CONF):
                continue

            left, top, box_width, box_height = self._xywh_to_ltwh(xywh, width, height)
            if box_width <= 0 or box_height <= 0:
                continue

            predictions.append(
                {
                    "bbox": [left, top, box_width, box_height],
                    "category_id": int(category_id),
                }
            )

        return predictions

    def _xywh_to_ltwh(
        self,
        xywh: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        center_x, center_y, box_width, box_height = (float(v) for v in xywh)
        left = center_x - box_width / 2
        top = center_y - box_height / 2
        right = center_x + box_width / 2
        bottom = center_y + box_height / 2

        left = int(round(max(0, min(left, image_width - 1))))
        top = int(round(max(0, min(top, image_height - 1))))
        right = int(round(max(0, min(right, image_width))))
        bottom = int(round(max(0, min(bottom, image_height))))

        return left, top, right - left, bottom - top
