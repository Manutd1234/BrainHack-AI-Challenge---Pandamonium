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
MODEL_PATHS = [
    path.strip()
    for path in os.getenv("CV_MODEL_PATHS", MODEL_PATH).split(",")
    if path.strip()
]
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
RAW_MAX_DETECTIONS = int(os.getenv("CV_RAW_MAX_DETECTIONS", max(MAX_DETECTIONS * 4, 50)))
PRED_BATCH_SIZE = int(os.getenv("CV_PRED_BATCH_SIZE", "4"))
USE_AUGMENT = os.getenv("CV_AUGMENT", "0").strip().lower() in {"1", "true", "yes"}
FINAL_NMS_IOU = float(os.getenv("CV_FINAL_NMS_IOU", "0.55"))
USE_WBF = os.getenv("CV_USE_WBF", "1").strip().lower() in {"1", "true", "yes"}
FALLBACK_FLIP = os.getenv("CV_FALLBACK_FLIP", "0").strip().lower() in {"1", "true", "yes"}
FALLBACK_MAX_COUNT = int(os.getenv("CV_FALLBACK_MAX_COUNT", "0"))
FALLBACK_MIN_CONF = float(os.getenv("CV_FALLBACK_MIN_CONF", "0.35"))
HIGHRES_FALLBACK = os.getenv("CV_HIGHRES_FALLBACK", "1").strip().lower() in {"1", "true", "yes"}
HIGHRES_IMGSZ = int(os.getenv("CV_HIGHRES_IMGSZ", "1536"))
HIGHRES_MAX_COUNT = int(os.getenv("CV_HIGHRES_MAX_COUNT", "0"))
HIGHRES_MIN_CONF = float(os.getenv("CV_HIGHRES_MIN_CONF", "0.45"))
USE_SAHI = os.getenv("CV_USE_SAHI", "0").strip().lower() in {"1", "true", "yes"}
SAHI_MIN_SIZE = int(os.getenv("CV_SAHI_MIN_SIZE", "640"))
SAHI_SLICE_SIZE = int(os.getenv("CV_SAHI_SLICE_SIZE", "640"))
SAHI_OVERLAP = float(os.getenv("CV_SAHI_OVERLAP", "0.20"))
SAHI_NMS_IOU = float(os.getenv("CV_SAHI_NMS_IOU", "0.50"))

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
PREDICT_CONF = float(os.getenv("CV_MODEL_CONF", min([DEFAULT_CONF, *CLASS_CONF.values()])))


class CVManager:
    """Loads a YOLO26x checkpoint and returns TIL-format detections."""

    def __init__(self) -> None:
        existing_model_paths = [path for path in MODEL_PATHS if os.path.exists(path)]
        if not existing_model_paths:
            raise FileNotFoundError(
                f"CV checkpoint not found at {MODEL_PATHS}. Run cv_train.py and "
                "copy model/best.pt into cv/model/best.pt before building."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = []
        for model_path in existing_model_paths:
            LOGGER.info("Loading YOLO checkpoint from %s on %s", model_path, self.device)
            model = YOLO(model_path)
            model.to(self.device)
            self.models.append(model)
        self.model = self.models[0]

        self.sahi_model = self._load_sahi_model() if USE_SAHI else None
        for model in self.models:
            model.predict(
                np.zeros((640, 640, 3), dtype=np.uint8),
                imgsz=DEFAULT_IMGSZ,
                conf=PREDICT_CONF,
                iou=DEFAULT_IOU,
                half=self.device == "cuda",
                augment=USE_AUGMENT,
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
        predictions = []
        batch_size = max(1, PRED_BATCH_SIZE)
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            batch_predictions: list[list[dict[str, Any]]] = [[] for _ in batch]
            batch_confidences: list[list[float]] = [[] for _ in batch]

            for model in self.models:
                results = model.predict(
                    batch,
                    conf=PREDICT_CONF,
                    iou=DEFAULT_IOU,
                    imgsz=DEFAULT_IMGSZ,
                    half=self.device == "cuda",
                    augment=USE_AUGMENT,
                    max_det=RAW_MAX_DETECTIONS,
                    verbose=False,
                    device=self.device,
                )
                for index, (result, image) in enumerate(zip(results, batch)):
                    formatted, confidences = self._format_yolo_result(result, image.shape)
                    batch_predictions[index].extend(formatted)
                    batch_confidences[index].extend(confidences)

            if FALLBACK_FLIP:
                for index, image in enumerate(batch):
                    if self._needs_flip_fallback(batch_predictions[index], batch_confidences[index]):
                        formatted, confidences = self._predict_horizontal_flip(image)
                        batch_predictions[index].extend(formatted)
                        batch_confidences[index].extend(confidences)

            if HIGHRES_FALLBACK and HIGHRES_IMGSZ > DEFAULT_IMGSZ:
                for index, image in enumerate(batch):
                    if self._needs_highres_fallback(batch_predictions[index], batch_confidences[index]):
                        formatted, confidences = self._predict_highres(image)
                        batch_predictions[index].extend(formatted)
                        batch_confidences[index].extend(confidences)

            predictions.extend(
                self._postprocess_predictions(single_predictions, single_confidences)
                for single_predictions, single_confidences in zip(
                    batch_predictions, batch_confidences
                )
            )
        return predictions

    def _needs_flip_fallback(
        self,
        predictions: list[dict[str, Any]],
        confidences: list[float],
    ) -> bool:
        if len(predictions) <= FALLBACK_MAX_COUNT:
            return True
        return bool(confidences and max(confidences) < FALLBACK_MIN_CONF)

    def _needs_highres_fallback(
        self,
        predictions: list[dict[str, Any]],
        confidences: list[float],
    ) -> bool:
        if len(predictions) <= HIGHRES_MAX_COUNT:
            return True
        return bool(confidences and max(confidences) < HIGHRES_MIN_CONF)

    def _predict_highres(self, image: np.ndarray) -> tuple[list[dict[str, Any]], list[float]]:
        predictions: list[dict[str, Any]] = []
        kept_confidences: list[float] = []

        for model in self.models:
            results = model.predict(
                [image],
                conf=PREDICT_CONF,
                iou=DEFAULT_IOU,
                imgsz=HIGHRES_IMGSZ,
                half=self.device == "cuda",
                augment=False,
                max_det=RAW_MAX_DETECTIONS,
                verbose=False,
                device=self.device,
            )
            formatted, confidences = self._format_yolo_result(results[0], image.shape)
            predictions.extend(formatted)
            kept_confidences.extend(confidences)

        return predictions, kept_confidences

    def _predict_horizontal_flip(self, image: np.ndarray) -> tuple[list[dict[str, Any]], list[float]]:
        flipped = cv2.flip(image, 1)
        height, width = image.shape[:2]
        predictions: list[dict[str, Any]] = []
        kept_confidences: list[float] = []

        for model in self.models:
            results = model.predict(
                [flipped],
                conf=max(0.01, PREDICT_CONF * 0.75),
                iou=DEFAULT_IOU,
                imgsz=DEFAULT_IMGSZ,
                half=self.device == "cuda",
                augment=False,
                max_det=RAW_MAX_DETECTIONS,
                verbose=False,
                device=self.device,
            )
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes_xywh = result.boxes.xywh.detach().cpu().numpy()
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            for xywh, category_id, confidence in zip(boxes_xywh, class_ids, confidences):
                if category_id < 0 or category_id >= len(CLASS_NAMES):
                    continue
                if confidence < CLASS_CONF.get(int(category_id), DEFAULT_CONF):
                    continue
                unflipped_xywh = np.array(xywh, dtype=np.float32)
                unflipped_xywh[0] = float(width) - float(unflipped_xywh[0])
                left, top, box_width, box_height = self._xywh_to_ltwh(
                    unflipped_xywh,
                    width,
                    height,
                )
                if box_width <= 0 or box_height <= 0:
                    continue
                predictions.append(
                    {
                        "bbox": [left, top, box_width, box_height],
                        "category_id": int(category_id),
                        "_confidence": float(confidence),
                    }
                )
                kept_confidences.append(float(confidence))

        return predictions, kept_confidences

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
            postprocess_match_threshold=SAHI_NMS_IOU,
            postprocess_class_agnostic=False,
            verbose=0,
        )

        predictions = []
        confidences = []
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
                    "_confidence": float(obj.score.value),
                }
            )
            confidences.append(float(obj.score.value))

        return self._postprocess_predictions(predictions, confidences)

    def _format_yolo_result(
        self,
        result,
        image_shape: tuple[int, int, int],
    ) -> tuple[list[dict[str, Any]], list[float]]:
        if result.boxes is None or len(result.boxes) == 0:
            return [], []

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
                    "_confidence": float(confidence),
                }
            )
            kept_confidences.append(float(confidence))

        return predictions, kept_confidences

    def _postprocess_predictions(
        self,
        predictions: list[dict[str, Any]],
        confidences: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if confidences is None:
            confidences = [float(prediction.get("_confidence", 1.0)) for prediction in predictions]

        predictions = self._classwise_nms(predictions, confidences)
        confidences = [float(prediction.get("_confidence", 1.0)) for prediction in predictions]
        if len(predictions) > MAX_DETECTIONS:
            order = np.argsort(-np.asarray(confidences))[:MAX_DETECTIONS]
            predictions = [predictions[int(index)] for index in order]

        for prediction in predictions:
            prediction.pop("_confidence", None)
        return predictions

    def _classwise_nms(
        self,
        predictions: list[dict[str, Any]],
        confidences: list[float],
    ) -> list[dict[str, Any]]:
        if not predictions:
            return []
        if USE_WBF:
            return self._classwise_weighted_fusion(predictions, confidences)

        kept: list[dict[str, Any]] = []
        by_class: dict[int, list[int]] = {}
        for index, prediction in enumerate(predictions):
            by_class.setdefault(int(prediction["category_id"]), []).append(index)

        for indices in by_class.values():
            ordered = sorted(indices, key=lambda index: confidences[index], reverse=True)
            while ordered:
                current = ordered.pop(0)
                kept.append(predictions[current])
                ordered = [
                    other
                    for other in ordered
                    if self._bbox_iou(predictions[current]["bbox"], predictions[other]["bbox"])
                    < FINAL_NMS_IOU
                ]

        kept.sort(key=lambda prediction: float(prediction.get("_confidence", 1.0)), reverse=True)
        return kept

    def _classwise_weighted_fusion(
        self,
        predictions: list[dict[str, Any]],
        confidences: list[float],
    ) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        by_class: dict[int, list[int]] = {}
        for index, prediction in enumerate(predictions):
            by_class.setdefault(int(prediction["category_id"]), []).append(index)

        for category_id, indices in by_class.items():
            ordered = sorted(indices, key=lambda index: confidences[index], reverse=True)
            while ordered:
                current = ordered.pop(0)
                group = [current]
                remaining = []
                for other in ordered:
                    if self._bbox_iou(predictions[current]["bbox"], predictions[other]["bbox"]) >= FINAL_NMS_IOU:
                        group.append(other)
                    else:
                        remaining.append(other)
                ordered = remaining

                weights = np.asarray(
                    [max(1e-6, confidences[index]) for index in group],
                    dtype=np.float32,
                )
                boxes = np.asarray([predictions[index]["bbox"] for index in group], dtype=np.float32)
                fused_box = np.average(boxes, axis=0, weights=weights).tolist()
                kept.append(
                    {
                        "bbox": [float(value) for value in fused_box],
                        "category_id": int(category_id),
                        "_confidence": float(max(confidences[index] for index in group)),
                    }
                )

        kept.sort(key=lambda prediction: float(prediction.get("_confidence", 1.0)), reverse=True)
        return kept

    def _bbox_iou(self, first: list[float], second: list[float]) -> float:
        first_x1, first_y1, first_w, first_h = first
        second_x1, second_y1, second_w, second_h = second
        first_x2 = first_x1 + first_w
        first_y2 = first_y1 + first_h
        second_x2 = second_x1 + second_w
        second_y2 = second_y1 + second_h

        inter_x1 = max(first_x1, second_x1)
        inter_y1 = max(first_y1, second_y1)
        inter_x2 = min(first_x2, second_x2)
        inter_y2 = min(first_y2, second_y2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        union = first_w * first_h + second_w * second_h - intersection
        return intersection / union if union > 0 else 0.0

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
