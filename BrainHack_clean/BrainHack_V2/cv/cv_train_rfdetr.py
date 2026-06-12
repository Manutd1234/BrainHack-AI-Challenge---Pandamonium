"""Optional RF-DETR fine-tuning for the TIL-AI 2026 CV task.

This is a candidate path, not the baseline. It converts the public novice
COCO annotations into the directory shapes commonly expected by RF-DETR, then
copies the best checkpoint to model/rfdetr_best.pt for optional ensemble use.

Run in Jupyter:

    cd ~/cv
    RFDETR_EPOCHS=40 RFDETR_BATCH=4 python cv_train_rfdetr.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image


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
OUT_DIR = Path(os.getenv("RFDETR_DATA_DIR", "/home/jupyter/cv/data/rfdetr_til26"))


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


def reset_dir(path: Path) -> None:
    """Recreate a directory, tolerating slow/stale notebook filesystems."""
    if path.exists():
        for attempt in range(3):
            shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                break
            time.sleep(0.4 * (attempt + 1))

    if path.exists():
        # Last resort: clear children one by one. Some mounted filesystems can
        # report "directory not empty" even after rmtree has walked the tree.
        for child in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            try:
                if child.is_dir() and not child.is_symlink():
                    child.rmdir()
                else:
                    child.unlink()
            except OSError:
                pass
        try:
            path.rmdir()
        except OSError:
            pass

    path.mkdir(parents=True, exist_ok=True)


def prepare_dataset() -> Path:
    images_dir = NOVICE_CV_DIR / "images"
    ann_path = NOVICE_CV_DIR / "annotations.json"
    source = json.loads(ann_path.read_text(encoding="utf-8"))
    images: list[dict[str, Any]] = source["images"]
    annotations: list[dict[str, Any]] = source["annotations"]

    cat_ids = sorted({int(annotation["category_id"]) for annotation in annotations})
    cat_to_contiguous = {category_id: index for index, category_id in enumerate(cat_ids)}

    anns_by_image: dict[Any, list[dict[str, Any]]] = {}
    for annotation in annotations:
        anns_by_image.setdefault(annotation["image_id"], []).append(annotation)

    shuffled = list(images)
    random.seed(26)
    random.shuffle(shuffled)
    cut = int(len(shuffled) * float(os.getenv("RFDETR_TRAIN_FRAC", "0.9")))
    splits = {"train": shuffled[:cut], "valid": shuffled[cut:]}

    for split in splits:
        split_dir = OUT_DIR / split
        reset_dir(split_dir)

    for split, split_images in splits.items():
        write_split(split, split_images, images_dir, anns_by_image, cat_to_contiguous)

    print(f"RF-DETR train images: {len(splits['train'])}")
    print(f"RF-DETR valid images: {len(splits['valid'])}")
    print(f"RF-DETR dataset written to {OUT_DIR}")
    return OUT_DIR


def write_split(
    split: str,
    images: list[dict[str, Any]],
    images_dir: Path,
    anns_by_image: dict[Any, list[dict[str, Any]]],
    cat_to_contiguous: dict[int, int],
) -> None:
    split_dir = OUT_DIR / split
    coco_images = []
    coco_annotations = []
    annotation_id = 1

    for image in images:
        name = str(image["file_name"])
        src_img = images_dir / name
        if not src_img.exists():
            continue

        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if not width or not height:
            with Image.open(src_img) as handle:
                width, height = handle.size

        link_or_copy(src_img, split_dir / name)
        image_id = int(image["id"])
        coco_images.append(
            {
                "id": image_id,
                "file_name": name,
                "width": width,
                "height": height,
            }
        )

        for annotation in anns_by_image.get(image["id"], []):
            x, y, box_width, box_height = (
                float(value) for value in annotation["bbox"]
            )
            category_id = cat_to_contiguous[int(annotation["category_id"])]
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, box_width, box_height],
                    "area": max(0.0, box_width * box_height),
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                }
            )
            annotation_id += 1

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": index, "name": name}
            for index, name in enumerate(TIL_CLASSES)
        ],
    }
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(coco),
        encoding="utf-8",
    )


def train() -> None:
    dataset_dir = prepare_dataset()
    from rfdetr import RFDETRLarge

    epochs = int(os.getenv("RFDETR_EPOCHS", "40"))
    batch_size = int(os.getenv("RFDETR_BATCH", "4"))
    grad_accum = int(os.getenv("RFDETR_GRAD_ACCUM", "2"))
    resolution = int(os.getenv("RFDETR_RESOLUTION", "800"))
    output_dir = os.getenv("RFDETR_OUTPUT_DIR", "runs/rfdetr_til26")

    model_kwargs = {
        "num_classes": len(TIL_CLASSES),
        "resolution": resolution,
    }
    pretrain_weights = os.getenv("RFDETR_PRETRAIN_WEIGHTS", "").strip()
    if pretrain_weights:
        model_kwargs["pretrain_weights"] = pretrain_weights

    try:
        model = RFDETRLarge(**model_kwargs)
    except Exception as exc:
        print(f"RFDETRLarge({model_kwargs}) failed: {exc}")
        print("Retrying with RFDETRLarge() defaults.")
        model = RFDETRLarge()

    train_kwargs = {
        "dataset_dir": str(dataset_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "resolution": resolution,
        "output_dir": output_dir,
        "num_workers": int(os.getenv("RFDETR_WORKERS", "4")),
    }
    try:
        model.train(**train_kwargs, grad_accum_steps=grad_accum)
    except TypeError as exc:
        if "grad_accum_steps" not in str(exc):
            raise
        model.train(**train_kwargs, grad_accumulation_steps=grad_accum)

    candidates = [
        Path(output_dir) / "checkpoint_best.pth",
        Path(output_dir) / "best.pth",
        Path(output_dir) / "best.pt",
    ]
    best = next((candidate for candidate in candidates if candidate.exists()), None)
    if best is None:
        best = next(Path(output_dir).rglob("*best*"), None)
    if best is None or not best.exists():
        raise FileNotFoundError(f"Could not find RF-DETR best checkpoint in {output_dir}")

    Path("model").mkdir(exist_ok=True)
    shutil.copy(best, "model/rfdetr_best.pt")
    print(f"RF-DETR candidate copied from {best} to model/rfdetr_best.pt")


if __name__ == "__main__":
    train()
