"""YOLO26x CV manager for TIL-AI 2026."""

from __future__ import annotations

import logging
import os
import json
from typing import Any
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)
MODEL_PATH = os.getenv("CV_MODEL_PATH", "/app/model/best.pt")
THRESHOLD_CONFIG_PATH = Path(
    os.getenv("CV_THRESHOLD_CONFIG", Path(__file__).with_name("cv_thresholds.json"))
)


def _load_threshold_config() -> dict[str, Any]:
    if not THRESHOLD_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(THRESHOLD_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Could not load CV threshold config %s: %s", THRESHOLD_CONFIG_PATH, exc)
        return {}


THRESHOLD_CONFIG = _load_threshold_config()
DEFAULT_CONF = float(os.getenv("CV_DEFAULT_CONF", THRESHOLD_CONFIG.get("default_conf", 0.20)))
DEFAULT_IOU = float(os.getenv("CV_IOU", THRESHOLD_CONFIG.get("iou", 0.45)))
DEFAULT_IMGSZ = int(os.getenv("CV_IMGSZ", THRESHOLD_CONFIG.get("imgsz", 1280)))
MAX_DETECTIONS = int(os.getenv("CV_MAX_DETECTIONS", THRESHOLD_CONFIG.get("max_detections", 20)))
USE_SAHI = os.getenv("CV_USE_SAHI", "0").strip().lower() in {"1", "true", "yes"}
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

DEFAULT_CLASS_CONF = {
    0: 0.35,
    1: 0.35,
    2: 0.28,
    3: 0.35,
    4: 0.35,
    5: 0.32,
    6: 0.32,
    7: 0.28,
    8: 0.35,
    9: 0.35,
    10: 0.35,
    11: 0.35,
    12: 0.35,
    13: 0.35,
    14: 0.32,
    15: 0.35,
    16: 0.35,
    17: 0.32,
}
CLASS_CONF = {
    int(category_id): float(confidence)
    for category_id, confidence in THRESHOLD_CONFIG.get("class_conf", DEFAULT_CLASS_CONF).items()
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

        self.sahi_model = self._load_sahi_model() if USE_SAHI else None
        self.model.predict(
            np.zeros((640, 640, 3), dtype=np.uint8),
            imgsz=DEFAULT_IMGSZ,
            conf=DEFAULT_CONF,
            iou=DEFAULT_IOU,
            half=self.device == "cuda",
            verbose=False,
        )
        LOGGER.info("CVManager ready")

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Perform object detection on one JPEG image."""
        return self.cv_many([image])[0]

    def cv_many(self, images: list[bytes]) -> list[list[dict[str, Any]]]:
        """Perform object detection on a request batch."""
        frames = [self._decode(image) for image in images]
        if not frames:
            return []

        if self.sahi_model is None:
            return self._predict_full_batch(frames)

        predictions = []
        for frame in frames:
            height, width = frame.shape[:2]
            if max(height, width) >= SAHI_MIN_SIZE:
                predictions.append(self._predict_sahi(frame))
            else:
                predictions.extend(self._predict_full_batch([frame]))
        return predictions

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
        return self._predict_full_batch([image])[0]

    def _predict_full_batch(self, images: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        results = self.model.predict(
            images,
            conf=DEFAULT_CONF,
            iou=DEFAULT_IOU,
            imgsz=DEFAULT_IMGSZ,
            half=self.device == "cuda",
            max_det=MAX_DETECTIONS,
            verbose=False,
            device=self.device,
        )
        return [
            self._format_yolo_result(result, image.shape)
            for result, image in zip(results, images)
        ]

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

            left = float(max(0.0, min(float(obj.bbox.minx), float(width - 1))))
            top = float(max(0.0, min(float(obj.bbox.miny), float(height - 1))))
            right = float(max(0.0, min(float(obj.bbox.maxx), float(width))))
            bottom = float(max(0.0, min(float(obj.bbox.maxy), float(height))))
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

        return self._limit_predictions(predictions)

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
        kept_confidences = []
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
            kept_confidences.append(float(confidence))

        return self._limit_predictions(predictions, kept_confidences)

    def _limit_predictions(
        self,
        predictions: list[dict[str, Any]],
        confidences: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if len(predictions) <= MAX_DETECTIONS:
            return predictions
        if confidences is None:
            return predictions[:MAX_DETECTIONS]
        order = np.argsort(-np.asarray(confidences))[:MAX_DETECTIONS]
        return [predictions[int(index)] for index in order]

    def _xywh_to_ltwh(
        self,
        xywh: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        center_x, center_y, box_width, box_height = (float(v) for v in xywh)
        left = center_x - box_width / 2
        top = center_y - box_height / 2
        right = center_x + box_width / 2
        bottom = center_y + box_height / 2

        left = float(max(0.0, min(left, float(image_width - 1))))
        top = float(max(0.0, min(top, float(image_height - 1))))
        right = float(max(0.0, min(right, float(image_width))))
        bottom = float(max(0.0, min(bottom, float(image_height))))

        return left, top, right - left, bottom - top
