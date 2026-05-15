"""Fine-tune YOLO26x for the TIL-AI 2026 CV challenge."""

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
    """Write the local Ultralytics dataset config."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(DATA_YAML, handle, default_flow_style=False, sort_keys=False)
    print(f"data.yaml written to {path}")
    return path


def train() -> None:
    """Train YOLO26x and copy the best checkpoint into model/best.pt."""
    data_yaml = write_data_yaml()
    model = YOLO("yolo26x.pt")

    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=1280,
        batch=4,
        device="cuda",
        workers=4,
        project="runs",
        name="yolo26x_til26",
        exist_ok=True,
        degrees=90.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )

    os.makedirs("model", exist_ok=True)
    shutil.copy("runs/yolo26x_til26/weights/best.pt", "model/best.pt")
    metric = results.results_dict.get("metrics/mAP50-95(B)", "N/A")
    print(f"mAP50-95: {metric}")
    print("Checkpoint saved to model/best.pt")


if __name__ == "__main__":
    train()
