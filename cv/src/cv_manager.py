"""Manages the CV model."""

import io
import logging
from typing import Any

from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class CVManager:

    def __init__(self):
        # Load YOLO model. yolo11n.pt was pre-downloaded during Docker build.
        logger.info("Loading YOLO model...")
        self.model = YOLO("yolo11n.pt")
        logger.info("YOLO model loaded.")

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes (JPEG format).

        Returns:
            A list of `dict`s, each with keys:
              - "bbox": [x, y, w, h] (top-left corner, width, height)
              - "category_id": int (class index)
        """

        img = Image.open(io.BytesIO(image))

        try:
            results = self.model.predict(img, verbose=False)
        except Exception:
            # Fallback to CPU if GPU fails
            self.model = YOLO("yolo11n.pt")
            results = self.model.predict(img, verbose=False, device="cpu")

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # YOLO returns xyxy format; convert to xywh
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                category_id = int(box.cls[0].item())

                detections.append({
                    "bbox": [x1, y1, w, h],
                    "category_id": category_id,
                })

        return detections
