"""Final CV fine-tuning pass on all novice images.

Use this after the first strong YOLO26x checkpoint exists at model/best.pt.
It rebuilds a YOLO-format dataset from /home/jupyter/novice/cv, trains on all
public novice images with conservative augmentation, and writes a new candidate
checkpoint to model/best.pt.

Run in Jupyter:

    cd ~/cv
    CV_FINAL_EPOCHS=18 CV_FINAL_IMGSZ=1280 CV_FINAL_BATCH=2 python cv_train_final.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
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

NOVICE_CV_DIR = Path(os.getenv("CV_NOVICE_DIR", "/home/jupyter/novice/cv"))
OUT_DIR = Path(os.getenv("CV_FINAL_DATA_DIR", "/home/jupyter/cv/data/til26_final"))


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def prepare_dataset() -> str:
    images_dir = NOVICE_CV_DIR / "images"
    ann_path = NOVICE_CV_DIR / "annotations.json"
    annotations_json = json.loads(ann_path.read_text(encoding="utf-8"))
    images: list[dict[str, Any]] = annotations_json["images"]
    annotations: list[dict[str, Any]] = annotations_json["annotations"]

    category_ids = sorted({int(annotation["category_id"]) for annotation in annotations})
    cat_to_yolo = {category_id: index for index, category_id in enumerate(category_ids)}

    anns_by_image: dict[Any, list[dict[str, Any]]] = {}
    for annotation in annotations:
        anns_by_image.setdefault(annotation["image_id"], []).append(annotation)

    random.seed(26)
    val_images = list(images)
    random.shuffle(val_images)
    val_images = val_images[int(len(val_images) * 0.9) :]

    for subdir in ("images/train", "images/val", "labels/train", "labels/val"):
        target = OUT_DIR / subdir
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    # Train on every public image. Keep a deterministic validation subset only
    # for Ultralytics bookkeeping and sanity checks.
    write_split("train", images, images_dir, anns_by_image, cat_to_yolo)
    write_split("val", val_images, images_dir, anns_by_image, cat_to_yolo)

    data_yaml = {
        "path": str(OUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(TIL_CLASSES)},
        "nc": len(TIL_CLASSES),
    }
    yaml_path = Path("data/til26_final.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.safe_dump(data_yaml, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    print(f"final train images: {len(list((OUT_DIR / 'images/train').glob('*')))}")
    print(f"final val images: {len(list((OUT_DIR / 'images/val').glob('*')))}")
    print(f"data.yaml written to {yaml_path}")
    return str(yaml_path)


def write_split(
    split: str,
    images: list[dict[str, Any]],
    images_dir: Path,
    anns_by_image: dict[Any, list[dict[str, Any]]],
    cat_to_yolo: dict[int, int],
) -> None:
    for image in images:
        name = str(image["file_name"])
        src_img = images_dir / name
        if not src_img.exists():
            continue
        link_or_copy(src_img, OUT_DIR / f"images/{split}" / name)

        width = image.get("width")
        height = image.get("height")
        if not width or not height:
            with Image.open(src_img) as handle:
                width, height = handle.size

        lines = []
        for annotation in anns_by_image.get(image["id"], []):
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            cls = cat_to_yolo[int(annotation["category_id"])]
            x_center = (x + box_width / 2.0) / float(width)
            y_center = (y + box_height / 2.0) / float(height)
            rel_width = box_width / float(width)
            rel_height = box_height / float(height)
            lines.append(
                f"{cls} {x_center:.8f} {y_center:.8f} {rel_width:.8f} {rel_height:.8f}"
            )

        label_path = OUT_DIR / f"labels/{split}" / f"{Path(name).stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data_yaml = prepare_dataset()
    base_checkpoint = Path(os.getenv("CV_FINAL_BASE", "model/best.pt"))
    if not base_checkpoint.exists():
        raise FileNotFoundError(
            f"{base_checkpoint} not found. Copy your best baseline checkpoint there first."
        )

    epochs = int(os.getenv("CV_FINAL_EPOCHS", "18"))
    imgsz = int(os.getenv("CV_FINAL_IMGSZ", "1280"))
    batch = int(os.getenv("CV_FINAL_BATCH", "2"))
    lr0 = float(os.getenv("CV_FINAL_LR0", "0.0006"))
    name = os.getenv("CV_FINAL_NAME", "yolo26x_til26_final")

    model = YOLO(str(base_checkpoint))
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cuda",
        workers=4,
        project="runs",
        name=name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=lr0,
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=1,
        patience=8,
        degrees=12.0,
        translate=0.08,
        scale=0.45,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.4,
        mixup=0.05,
        copy_paste=0.05,
        erasing=0.12,
        hsv_h=0.01,
        hsv_s=0.40,
        hsv_v=0.30,
        close_mosaic=3,
        save=True,
        save_period=5,
        val=True,
        plots=True,
    )

    candidates = [
        Path("runs") / name / "weights" / "best.pt",
        Path("runs/detect/runs") / name / "weights" / "best.pt",
        Path("runs/detect") / name / "weights" / "best.pt",
    ]
    best = next((candidate for candidate in candidates if candidate.exists()), None)
    if best is None:
        best = next(Path("runs").rglob(f"{name}/weights/best.pt"), None)
    if best is None or not best.exists():
        raise FileNotFoundError(f"Could not find best.pt for run {name}")

    Path("model").mkdir(exist_ok=True)
    backup = Path("model/best.before_final.pt")
    if not backup.exists():
        shutil.copy(base_checkpoint, backup)
    shutil.copy(best, "model/best.final.pt")
    if os.getenv("CV_FINAL_COPY_TO_BEST", "1").strip().lower() in {"1", "true", "yes"}:
        shutil.copy(best, "model/best.pt")

    metric = results.results_dict.get("metrics/mAP50-95(B)", "N/A")
    print(f"final fine-tune mAP50-95: {metric}")
    print(f"Final candidate saved at {best}")
    print("Copied to model/best.final.pt and model/best.pt")
    print(f"Previous checkpoint backed up at {backup}")


if __name__ == "__main__":
    main()
