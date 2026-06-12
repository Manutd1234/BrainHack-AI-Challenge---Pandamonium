from __future__ import annotations

import argparse
import copy
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

NUM_CLASSES = 18
THRESH_GRID = np.arange(0.20, 0.71, 0.025)
DEFAULT_THR = 0.40


def run_inference(model: YOLO, image_dir: str, image_records: list[dict], imgsz: int, low_conf: float) -> list[tuple]:
    raw = []
    for rec in image_records:
        path = f"{image_dir}/{rec['file_name']}"
        result = model.predict(path, imgsz=imgsz, conf=low_conf, iou=0.7, max_det=300, half=True, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xywh = result.boxes.xywh.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        for (cx, cy, w, h), c, s in zip(xywh, cls, conf):
            raw.append((rec["id"], int(c), float(s), float(cx - w / 2), float(cy - h / 2), float(w), float(h)))
    return raw


def score_with_thresholds(raw: list[tuple], annotations: dict, thresholds: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, float]:
    dets = []
    for img_id, c, s, x, y, w, h in raw:
        if s >= thresholds[c]:
            dets.append({"image_id": img_id, "score": 1.0, "bbox": [x, y, w, h], "category_id": c + 1})

    if not dets:
        return np.zeros(NUM_CLASSES), 0.0

    gt = COCO()
    gt.dataset = copy.deepcopy(annotations)
    gt.createIndex()
    dt = gt.loadRes(dets)
    ev = COCOeval(gt, dt, "bbox")
    if not verbose:
        ev.params.verbose = False
    ev.evaluate()
    ev.accumulate()
    if verbose:
        ev.summarize()

    precision = ev.eval["precision"][:, :, :, 0, 2]
    per_class = np.zeros(NUM_CLASSES)
    for k in range(precision.shape[2]):
        slc = precision[:, :, k]
        per_class[k] = slc[slc > -1].mean() if (slc > -1).any() else 0.0
    return per_class, float(per_class.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--val_images", required=True)
    parser.add_argument("--val_anno", required=True)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--low_conf", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", default="cv_thresholds.json")
    args = parser.parse_args()

    with open(args.val_anno, encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations['images'])} val images, {len(annotations['annotations'])} annotations.")

    model = YOLO(args.weights)
    print("Running one-shot inference at low conf...")
    raw = run_inference(model, args.val_images, annotations["images"], args.imgsz, args.low_conf)
    print(f"Got {len(raw)} raw detections to filter.")

    best_thr = np.full(NUM_CLASSES, DEFAULT_THR)
    _, baseline = score_with_thresholds(raw, annotations, best_thr)
    print(f"Baseline mAP at uniform {DEFAULT_THR}: {baseline:.4f}")

    for round_idx in range(args.rounds):
        for c in range(NUM_CLASSES):
            best_ap_c, best_t_c = -1.0, best_thr[c]
            for t in THRESH_GRID:
                trial = best_thr.copy()
                trial[c] = t
                per_class, _ = score_with_thresholds(raw, annotations, trial)
                if per_class[c] > best_ap_c:
                    best_ap_c, best_t_c = per_class[c], float(t)
            best_thr[c] = best_t_c
            print(f"[round {round_idx + 1}/{args.rounds}] class {c:2d} thr={best_t_c:.3f} AP={best_ap_c:.4f}")

    print("\n=== FINAL ===")
    final_pc, final_map = score_with_thresholds(raw, annotations, best_thr, verbose=True)
    print(f"Per-class AP: {final_pc}")
    print(f"Mean AP@.5:.05:.95: {final_map:.4f}")
    print(f"Improvement over baseline: {final_map - baseline:+.4f}")

    out = {
        "imgsz": args.imgsz,
        "iou": 0.50,
        "default_conf": float(DEFAULT_THR),
        "max_detections": 300,
        "class_conf": {str(i): float(best_thr[i]) for i in range(NUM_CLASSES)},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
