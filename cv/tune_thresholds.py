"""Tune CV output thresholds against the novice CV validation split.

Run after training and before Docker build:

    cd ~/cv
    python tune_thresholds.py

It writes src/cv_thresholds.json, which cv_manager.py loads at runtime. This is
especially useful for the TIL evaluator because submitted CV predictions do not
carry usable confidence scores, so low-confidence false positives are expensive.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


DATA_DIR = Path(os.getenv("CV_DATA_DIR", "/home/jupyter/novice/cv"))
PREPARED_DIR = Path(os.getenv("CV_PREPARED_DIR", "/home/jupyter/cv/data/til26"))
MODEL_PATH = Path(os.getenv("CV_MODEL_PATH", "/home/jupyter/cv/model/best.pt"))
OUT_PATH = Path(os.getenv("CV_THRESHOLD_OUTPUT", "/home/jupyter/cv/src/cv_thresholds.json"))
CACHE_PATH = Path(os.getenv("CV_TUNE_CACHE", "/home/jupyter/cv/cv_tune_predictions.json"))

IMGSZ = int(os.getenv("CV_TUNE_IMGSZ", "1280"))
BATCH = int(os.getenv("CV_TUNE_BATCH", "1"))
PRED_CONF = float(os.getenv("CV_TUNE_PRED_CONF", "0.01"))
PRED_IOU = float(os.getenv("CV_TUNE_PRED_IOU", "0.70"))
PRED_MAX_DET = int(os.getenv("CV_TUNE_PRED_MAX_DET", "100"))

GLOBAL_GRID = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.46, 0.52]
CLASS_GRID = [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.46, 0.52, 0.60]
MAX_DET_GRID = [8, 12, 16, 20, 25, 30]


def load_annotations() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, int]]:
    annotations = json.loads((DATA_DIR / "annotations.json").read_text(encoding="utf-8"))
    images = annotations["images"]
    anns = annotations["annotations"]
    category_ids = sorted({int(ann["category_id"]) for ann in anns})
    cat_to_yolo = {category_id: index for index, category_id in enumerate(category_ids)}
    return images, anns, cat_to_yolo


def validation_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    val_dir = PREPARED_DIR / "images" / "val"
    if val_dir.exists():
        val_names = {path.name for path in val_dir.glob("*")}
        selected = [image for image in images if str(image["file_name"]) in val_names]
        if selected:
            return selected

    shuffled = list(images)
    random.seed(26)
    random.shuffle(shuffled)
    return shuffled[int(len(shuffled) * 0.9) :]


def build_coco_gt(
    val_images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    cat_to_yolo: dict[int, int],
) -> dict[str, Any]:
    val_ids = {image["id"] for image in val_images}
    return {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": image["id"],
                "file_name": image["file_name"],
                "width": image.get("width", 0),
                "height": image.get("height", 0),
            }
            for image in val_images
        ],
        "annotations": [
            {
                "id": index + 1,
                "image_id": ann["image_id"],
                "category_id": cat_to_yolo[int(ann["category_id"])],
                "bbox": ann["bbox"],
                "area": float(ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": 0,
            }
            for index, ann in enumerate(annotations)
            if ann["image_id"] in val_ids and int(ann["category_id"]) in cat_to_yolo
        ],
        "categories": [{"id": index, "name": str(index)} for index in range(len(cat_to_yolo))],
    }


def predict_once(val_images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    model = YOLO(str(MODEL_PATH))
    image_paths = [str(DATA_DIR / "images" / str(image["file_name"])) for image in val_images]
    image_ids = [image["id"] for image in val_images]
    predictions: list[dict[str, Any]] = []

    for start in range(0, len(image_paths), BATCH):
        batch_paths = image_paths[start : start + BATCH]
        batch_ids = image_ids[start : start + BATCH]
        results = model.predict(
            batch_paths,
            imgsz=IMGSZ,
            conf=PRED_CONF,
            iou=PRED_IOU,
            max_det=PRED_MAX_DET,
            half=True,
            device=0,
            verbose=False,
        )
        for image_id, result in zip(batch_ids, results):
            if result.boxes is None:
                continue
            boxes = result.boxes.xywh.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            for xywh, category_id, confidence in zip(boxes, classes, confidences):
                x_center, y_center, width, height = (float(value) for value in xywh)
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": [
                            x_center - width / 2.0,
                            y_center - height / 2.0,
                            width,
                            height,
                        ],
                        "score": float(confidence),
                    }
                )

    CACHE_PATH.write_text(json.dumps(predictions), encoding="utf-8")
    return predictions


def evaluate(
    coco_gt_json: dict[str, Any],
    raw_predictions: list[dict[str, Any]],
    thresholds: dict[int, float],
    max_detections: int,
) -> float:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    filtered_by_image: dict[int, list[dict[str, Any]]] = {}
    for prediction in raw_predictions:
        threshold = thresholds.get(int(prediction["category_id"]), 0.30)
        if float(prediction["score"]) < threshold:
            continue
        copied = dict(prediction)
        copied["_rank_score"] = float(prediction["score"])
        # Match the challenge behavior: returned detections do not include usable confidence.
        copied["score"] = 1.0
        filtered_by_image.setdefault(int(prediction["image_id"]), []).append(copied)

    filtered: list[dict[str, Any]] = []
    for image_predictions in filtered_by_image.values():
        image_predictions.sort(key=lambda item: item["_rank_score"], reverse=True)
        for prediction in image_predictions[:max_detections]:
            prediction.pop("_rank_score", None)
            filtered.append(prediction)

    if not filtered:
        return 0.0

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as gt_file:
        json.dump(coco_gt_json, gt_file)
        gt_path = gt_file.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as pred_file:
        json.dump(filtered, pred_file)
        pred_path = pred_file.name

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return float(coco_eval.stats[0])


def main() -> None:
    images, annotations, cat_to_yolo = load_annotations()
    val_images = validation_images(images)
    coco_gt_json = build_coco_gt(val_images, annotations, cat_to_yolo)
    raw_predictions = predict_once(val_images)

    best_score = -1.0
    best_thresholds: dict[int, float] = {}
    best_max_det = 20

    for threshold in GLOBAL_GRID:
        thresholds = {category_id: threshold for category_id in range(len(cat_to_yolo))}
        for max_det in MAX_DET_GRID:
            score = evaluate(coco_gt_json, raw_predictions, thresholds, max_det)
            if score > best_score:
                best_score = score
                best_thresholds = dict(thresholds)
                best_max_det = max_det
                print(f"best global score={score:.4f} conf={threshold:.2f} max_det={max_det}", flush=True)

    for _ in range(2):
        for category_id in range(len(cat_to_yolo)):
            local_best_score = best_score
            local_best_threshold = best_thresholds[category_id]
            for threshold in CLASS_GRID:
                thresholds = dict(best_thresholds)
                thresholds[category_id] = threshold
                score = evaluate(coco_gt_json, raw_predictions, thresholds, best_max_det)
                if score > local_best_score:
                    local_best_score = score
                    local_best_threshold = threshold
            best_thresholds[category_id] = local_best_threshold
            if local_best_score > best_score:
                best_score = local_best_score
                print(
                    f"best class score={best_score:.4f} class={category_id} "
                    f"conf={local_best_threshold:.2f}",
                    flush=True,
                )

    output = {
        "score": best_score,
        "imgsz": IMGSZ,
        "iou": 0.50,
        "default_conf": float(np.median(list(best_thresholds.values()))),
        "max_detections": best_max_det,
        "class_conf": {str(key): value for key, value in sorted(best_thresholds.items())},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
