"""YOLO26x + RF-DETR-large CV manager with TTA, SAHI, and WBF for TIL-AI 2026."""

from __future__ import annotations

import base64
import logging
import os
import sys
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2ltwh

LOGGER = logging.getLogger(__name__)

# Search in order of preference for the YOLO weights
YOLO_PATHS = [
    os.getenv("CV_YOLO_PATH", "/app/model/yolo_best.pt"),
    "/app/model/best.pt",
    "model/yolo_best.pt",
    "model/best.pt",
    "yolo26x.pt"
]

RFDETR_PATHS = [
    os.getenv("CV_RFDETR_PATH", "/app/model/rfdetr_best.pt"),
    "model/rfdetr_best.pt"
]

CLASSES = [
    "cargo aircraft", "commercial aircraft", "drone", "fighter jet", "fighter plane",
    "helicopter", "light aircraft", "missile", "truck", "car", "tank", "bus", "van",
    "cargo ship", "yacht", "cruise ship", "warship", "sailboat"
]

# Better per-class thresholds from the Claude solution
CLASS_CONF = {
    0: 0.25, 1: 0.25, 2: 0.12, 3: 0.25, 4: 0.25, 5: 0.18, 6: 0.18, 7: 0.12,
    8: 0.25, 9: 0.25, 10: 0.25, 11: 0.25, 12: 0.22, 13: 0.25, 14: 0.18, 15: 0.25, 16: 0.25, 17: 0.18
}

DEFAULT_CONF = 0.18
WBF_IOU_THR = 0.50
WBF_SKIP_THR = 0.12
YOLO_W = 0.40
RFDETR_W = 0.60


class CVManager:
    """Loads fine-tuned YOLO26x and RF-DETR-large models and performs ensembling."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = None
        self.rfdetr = None
        self.sahi_yolo = None

        self._load_yolo()
        self._load_rfdetr()
        self._load_sahi()

        # Warmup model
        if self.yolo:
            self.yolo.predict(
                np.zeros((640, 640, 3), dtype=np.uint8),
                verbose=False
            )
        LOGGER.info("CVManager successfully initialized and ready")

    def _load_yolo(self) -> None:
        yolo_path = next((path for path in YOLO_PATHS if os.path.exists(path)), None)
        if yolo_path is None:
            # Fallback to downloading or using local yolo26x.pt
            LOGGER.warning("Fine-tuned YOLO checkpoint not found at %s. Falling back to yolo26x.pt", YOLO_PATHS)
            yolo_path = "yolo26x.pt"

        LOGGER.info("Loading YOLO checkpoint from %s on %s", yolo_path, self.device)
        self.yolo = YOLO(yolo_path)
        self.yolo.to(self.device)
        LOGGER.info("YOLO model loaded successfully")

    def _load_rfdetr(self) -> None:
        rfdetr_path = next((path for path in RFDETR_PATHS if os.path.exists(path)), None)
        if rfdetr_path is None:
            LOGGER.warning("RF-DETR checkpoint missing — running in YOLO-only mode")
            return

        try:
            from rfdetr import RFDETRLarge
            LOGGER.info("Loading RF-DETR checkpoint from %s on %s", rfdetr_path, self.device)
            self.rfdetr = RFDETRLarge(pretrain_weights=rfdetr_path)
            self.rfdetr.model.to(self.device).eval()
            LOGGER.info("RF-DETR-large model loaded successfully")
        except Exception as exc:
            LOGGER.warning("RF-DETR load failed (%s) — running in YOLO-only mode", exc)

    def _load_sahi(self) -> None:
        # Load SAHI wrapper using the resolved YOLO path
        yolo_path = next((path for path in YOLO_PATHS if os.path.exists(path)), "yolo26x.pt")
        try:
            from sahi import AutoDetectionModel
            self.sahi_yolo = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=yolo_path,
                confidence_threshold=DEFAULT_CONF,
                device=self.device
            )
            LOGGER.info("SAHI model wrapper loaded successfully")
        except Exception as exc:
            LOGGER.warning("SAHI model wrapper initialization failed (%s)", exc)

    def cv(self, image_bytes: bytes) -> list[dict[str, Any]]:
        """Perform object detection on one image."""
        return self.cv_many([image_bytes])[0]

    def cv_many(self, images: list[bytes]) -> list[list[dict[str, Any]]]:
        """Perform batched object detection for the challenge evaluator."""
        predictions = []
        for image_bytes in images:
            try:
                frame = self._decode(image_bytes)
                h, w = frame.shape[:2]

                # YOLO Inference (with TTA or SAHI)
                yp = self._get_yolo(frame)

                # RF-DETR Inference
                rp = self._get_rfdetr(frame)

                # Ensemble using Weighted Box Fusion (WBF)
                if yp or rp:
                    fused = self._wbf(yp, rp, h, w)
                    predictions.append(fused)
                else:
                    predictions.append([])
            except Exception as exc:
                LOGGER.error("Failed to process image in batch: %s", exc)
                predictions.append([])
        return predictions

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes as JPEG")
        return image

    def _get_yolo(self, image: np.ndarray) -> list[dict[str, Any]]:
        h, w = image.shape[:2]
        if self.sahi_yolo and max(h, w) >= 640:
            return self._yolo_sahi(image)
        return self._yolo_full(image)

    def _yolo_sahi(self, image: np.ndarray) -> list[dict[str, Any]]:
        from sahi.predict import get_sliced_prediction
        res = get_sliced_prediction(
            image,
            self.sahi_yolo,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=True,
            postprocess_type="NMS",
            postprocess_match_threshold=0.45,
            verbose=0
        )
        preds = []
        for obj in res.object_prediction_list:
            cat = int(obj.category.id)
            if obj.score.value < CLASS_CONF.get(cat, DEFAULT_CONF):
                continue
            b = obj.bbox
            l = int(b.minx)
            t = int(b.miny)
            w = int(b.maxx - b.minx)
            h = int(b.maxy - b.miny)
            if w > 0 and h > 0:
                preds.append({
                    "bbox": [l, t, w, h],
                    "category_id": cat,
                    "score": float(obj.score.value)
                })
        return preds

    def _yolo_full(self, image: np.ndarray) -> list[dict[str, Any]]:
        # augment=True enables built-in TTA (multi-scale + flip) at inference
        res = self.yolo.predict(
            image,
            conf=DEFAULT_CONF,
            iou=0.45,
            verbose=False,
            device=self.device,
            augment=True
        )
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes = xywh2ltwh(r.boxes.xywh.cpu().numpy())  # Convert to LTWH
        clsids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        preds = []
        for box, cls, conf in zip(boxes, clsids, confs):
            if conf < CLASS_CONF.get(int(cls), DEFAULT_CONF):
                continue
            l, t, w, h = (int(v) for v in box.tolist())
            if w > 0 and h > 0:
                preds.append({
                    "bbox": [l, t, w, h],
                    "category_id": int(cls),
                    "score": float(conf)
                })
        return preds

    def _get_rfdetr(self, image: np.ndarray) -> list[dict[str, Any]]:
        if self.rfdetr is None:
            return []
        h, w = image.shape[:2]
        try:
            import PIL.Image
            pil = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            dets = self.rfdetr.predict(pil, threshold=DEFAULT_CONF)
            preds = []
            for xyxy, cls, conf in zip(dets.xyxy, dets.class_id, dets.confidence):
                cat = int(cls)
                if conf < CLASS_CONF.get(cat, DEFAULT_CONF):
                    continue
                x1 = int(xyxy[0] * w)
                y1 = int(xyxy[1] * h)
                bw = int((xyxy[2] - xyxy[0]) * w)
                bh = int((xyxy[3] - xyxy[1]) * h)
                if bw > 0 and bh > 0:
                    preds.append({
                        "bbox": [x1, y1, bw, bh],
                        "category_id": cat,
                        "score": float(conf)
                    })
            return preds
        except Exception as exc:
            LOGGER.warning("RF-DETR inference error: %s", exc)
            return []

    def _wbf(
        self,
        yolo_p: list[dict[str, Any]],
        rfdetr_p: list[dict[str, Any]],
        img_h: int,
        img_w: int
    ) -> list[dict[str, Any]]:
        # If no RF-DETR is active, skip ensembling and output YOLO predictions
        if not rfdetr_p:
            return [{"bbox": p["bbox"], "category_id": p["category_id"]} for p in yolo_p]

        from ensemble_boxes import weighted_boxes_fusion

        def to_norm(preds):
            boxes, scores, labels = [], [], []
            for p in preds:
                l, t, w, h = p["bbox"]
                x1 = max(0.0, l / img_w)
                y1 = max(0.0, t / img_h)
                x2 = min(1.0, (l + w) / img_w)
                y2 = min(1.0, (t + h) / img_h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(p["score"])
                    labels.append(float(p["category_id"]))
            return boxes, scores, labels

        ybx, ysc, ylb = to_norm(yolo_p)
        rbx, rsc, rlb = to_norm(rfdetr_p)

        if not ybx and not rbx:
            return []

        boxes_l, scores_l, labels_l, weights = [], [], [], []
        if ybx:
            boxes_l.append(ybx)
            scores_l.append(ysc)
            labels_l.append(ylb)
            weights.append(YOLO_W)
        if rbx:
            boxes_l.append(rbx)
            scores_l.append(rsc)
            labels_l.append(rlb)
            weights.append(RFDETR_W)

        fb, fs, fl = weighted_boxes_fusion(
            boxes_l,
            scores_l,
            labels_l,
            weights=weights,
            iou_thr=WBF_IOU_THR,
            skip_box_thr=WBF_SKIP_THR
        )

        out = []
        for box, score, label in zip(fb, fs, fl):
            cat = int(round(label))
            if score < CLASS_CONF.get(cat, DEFAULT_CONF):
                continue
            x1 = int(box[0] * img_w)
            y1 = int(box[1] * img_h)
            bw = int((box[2] - box[0]) * img_w)
            bh = int((box[3] - box[1]) * img_h)
            if bw > 0 and bh > 0:
                out.append({
                    "bbox": [x1, y1, bw, bh],
                    "category_id": cat
                })
        return out
