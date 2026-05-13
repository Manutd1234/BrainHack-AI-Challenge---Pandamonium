"""Manages the CV model: YOLO26 detection with SAHI slicing.

The preferred path is a fine-tuned TIL-26 checkpoint in ``cv/models/best.pt``.
When that is unavailable, the manager falls back to COCO-pretrained YOLO26 and
maps overlapping COCO classes into the TIL-26 label space so the endpoint keeps
working during smoke tests.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

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

COCO_TO_TIL26_MAP = {
    2: 9,   # car -> car
    4: 1,   # airplane -> commercial aircraft
    5: 11,  # bus -> bus
    7: 8,   # truck -> truck
    8: 13,  # boat -> cargo ship
}

INFER_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1536"))
CONF_THRESHOLD = float(os.getenv("YOLO_CONF", "0.10"))
MAX_DETECTIONS = int(os.getenv("YOLO_MAX_DET", "200"))
SAHI_SLICE_SIZE = int(os.getenv("SAHI_SLICE_SIZE", "1024"))
SAHI_OVERLAP = float(os.getenv("SAHI_OVERLAP", "0.30"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _model_candidates() -> list[str]:
    configured = os.getenv("YOLO_MODEL_PATH")
    candidates = [
        configured,
        "models/best.pt",
        "best.pt",
        "models/yolo26x.pt",
        "models/yolo26l.pt",
        "models/yolo26m.pt",
        "yolo26x.pt",
        "yolo26l.pt",
        "yolo26m.pt",
        "yolo26s.pt",
    ]
    return [path for path in candidates if path]


def _predict_yolo(model: YOLO, image: Image.Image, imgsz: int):
    kwargs = {
        "verbose": False,
        "imgsz": imgsz,
        "conf": CONF_THRESHOLD,
        "max_det": MAX_DETECTIONS,
        "augment": _env_bool("CV_AUGMENT", False),
        "half": True,
        "end2end": True,
    }
    try:
        return model.predict(image, **kwargs)
    except TypeError:
        kwargs.pop("end2end", None)
        return model.predict(image, **kwargs)


class CVManager:
    def __init__(self):
        logger.info("Loading YOLO26 model...")

        self.model_path = self._resolve_model_path()
        self.model = YOLO(self.model_path)
        self.sahi_model = self._build_sahi_model(self.model_path)

        try:
            dummy = np.zeros((INFER_IMGSZ, INFER_IMGSZ, 3), dtype=np.uint8)
            _predict_yolo(self.model, Image.fromarray(dummy), imgsz=INFER_IMGSZ)
            logger.info("YOLO26 warm-up complete.")
        except Exception as exc:
            logger.warning("YOLO26 warm-up skipped: %s", exc)

        num_classes = len(getattr(self.model, "names", {}))
        self.is_finetuned = num_classes == len(TIL26_CLASSES)
        logger.info(
            "Loaded %s with %d classes. Fine-tuned for TIL-26: %s",
            self.model_path,
            num_classes,
            self.is_finetuned,
        )

    def _resolve_model_path(self) -> str:
        for model_path in _model_candidates():
            if os.path.exists(model_path):
                return model_path

        fallback = os.getenv("YOLO_FALLBACK_MODEL", "yolo26l.pt")
        logger.warning("No fine-tuned YOLO checkpoint found; using %s", fallback)
        YOLO(fallback)
        return fallback

    def _build_sahi_model(self, model_path: str):
        if not _env_bool("CV_USE_SAHI", True):
            return None
        try:
            from sahi import AutoDetectionModel

            device = "cuda:0"
            try:
                import torch

                if not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"

            return AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=model_path,
                confidence_threshold=CONF_THRESHOLD,
                device=device,
            )
        except Exception as exc:
            logger.warning("SAHI unavailable; using whole-image YOLO26: %s", exc)
            return None

    def _map_category(self, category_id: int) -> int | None:
        if self.is_finetuned:
            return category_id if category_id in TIL26_CLASSES else None
        return COCO_TO_TIL26_MAP.get(category_id)

    def _format_detection(
        self,
        xyxy: tuple[float, float, float, float],
        category_id: int,
        image_size: tuple[int, int],
        score: float | None = None,
    ) -> dict[str, Any] | None:
        mapped_cls = self._map_category(category_id)
        if mapped_cls is None:
            return None

        width, height = image_size
        x1, y1, x2, y2 = xyxy
        x1 = float(np.clip(x1, 0, width))
        y1 = float(np.clip(y1, 0, height))
        x2 = float(np.clip(x2, 0, width))
        y2 = float(np.clip(y2, 0, height))
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        if box_w <= 1.0 or box_h <= 1.0:
            return None

        detection = {
            "bbox": [x1, y1, box_w, box_h],
            "category_id": mapped_cls,
        }
        if score is not None:
            detection["score"] = float(score)
        return detection

    def _cv_sahi(self, img: Image.Image) -> list[dict[str, Any]]:
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            img,
            self.sahi_model,
            slice_height=SAHI_SLICE_SIZE,
            slice_width=SAHI_SLICE_SIZE,
            overlap_height_ratio=SAHI_OVERLAP,
            overlap_width_ratio=SAHI_OVERLAP,
            perform_standard_pred=True,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.50,
            verbose=0,
        )

        detections: list[dict[str, Any]] = []
        for pred in result.object_prediction_list[:MAX_DETECTIONS]:
            bbox = pred.bbox.to_xyxy()
            category_id = int(pred.category.id)
            score = getattr(getattr(pred, "score", None), "value", None)
            detection = self._format_detection(
                tuple(bbox),
                category_id,
                img.size,
                score=score,
            )
            if detection is not None:
                detections.append(detection)
        return detections

    def _cv_direct(self, img: Image.Image) -> list[dict[str, Any]]:
        results = _predict_yolo(self.model, img, imgsz=INFER_IMGSZ)
        detections: list[dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None

            for i in range(len(xyxy)):
                detection = self._format_detection(
                    tuple(float(v) for v in xyxy[i]),
                    int(cls[i]),
                    img.size,
                    score=float(conf[i]) if conf is not None else None,
                )
                if detection is not None:
                    detections.append(detection)
                if len(detections) >= MAX_DETECTIONS:
                    return detections

        return detections

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image."""
        img = Image.open(io.BytesIO(image)).convert("RGB")

        use_sahi = self.sahi_model is not None and (
            max(img.size) > SAHI_SLICE_SIZE or _env_bool("CV_ALWAYS_SAHI", True)
        )
        if use_sahi:
            try:
                return self._cv_sahi(img)
            except Exception as exc:
                logger.warning("SAHI prediction failed; falling back: %s", exc)

        return self._cv_direct(img)
