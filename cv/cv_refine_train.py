"""Refine an existing YOLO26x checkpoint for tighter CV boxes.

This is intended after the first full `cv_train.py` run has produced
`model/best.pt`. It continues from that checkpoint with a smaller learning rate,
larger image size, and lighter augmentation. That usually improves localization
precision more than another aggressive full-training run.

Run in Jupyter:

    cd ~/cv
    CV_REFINE_EPOCHS=40 CV_REFINE_IMGSZ=1536 CV_REFINE_BATCH=2 python cv_refine_train.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


TIL_CLASSES = [
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

DATA_YAML = {
    "path": str(Path("data/til26").resolve()),
    "train": "images/train",
    "val": "images/val",
    "names": {index: name for index, name in enumerate(TIL_CLASSES)},
    "nc": len(TIL_CLASSES),
}


def write_data_yaml(path: str = "data/til26.yaml") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(DATA_YAML, handle, default_flow_style=False, sort_keys=False)
    return path


def main() -> None:
    data_yaml = write_data_yaml()
    base_checkpoint = Path(os.getenv("CV_REFINE_BASE", "model/best.pt"))
    if not base_checkpoint.exists():
        raise FileNotFoundError(
            f"{base_checkpoint} not found. Run cv_train.py first or copy your best.pt there."
        )

    epochs = int(os.getenv("CV_REFINE_EPOCHS", "40"))
    imgsz = int(os.getenv("CV_REFINE_IMGSZ", "1536"))
    batch = int(os.getenv("CV_REFINE_BATCH", "2"))
    lr0 = float(os.getenv("CV_REFINE_LR0", "0.0012"))

    model = YOLO(str(base_checkpoint))
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cuda",
        workers=4,
        project="runs",
        name="yolo26x_til26_refine",
        exist_ok=True,
        optimizer="AdamW",
        lr0=lr0,
        lrf=0.02,
        cos_lr=True,
        warmup_epochs=1,
        patience=15,
        degrees=8.0,
        translate=0.05,
        scale=0.35,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.15,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.05,
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.25,
        close_mosaic=5,
        save=True,
        save_period=5,
        val=True,
        plots=True,
    )

    best = Path("runs/yolo26x_til26_refine/weights/best.pt")
    if not best.exists():
        raise FileNotFoundError(f"Expected refined checkpoint at {best}")

    backup = Path("model/best.before_refine.pt")
    if not backup.exists():
        shutil.copy(base_checkpoint, backup)
    shutil.copy(best, "model/best.pt")

    metric = results.results_dict.get("metrics/mAP50-95(B)", "N/A")
    print(f"refined mAP50-95: {metric}")
    print("Refined checkpoint saved to model/best.pt")
    print(f"Previous checkpoint backed up at {backup}")


if __name__ == "__main__":
    main()
