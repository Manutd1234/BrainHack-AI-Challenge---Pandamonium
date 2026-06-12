"""Utility to prepare YOLO and COCO formatted datasets for the CV challenge.

Reads COCO annotations from /home/jupyter/novice/cv and builds a dual YOLO
and COCO structured dataset under data/til26.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from PIL import Image


NOVICE_CV_DIR = Path(os.getenv("CV_NOVICE_DIR", "/home/jupyter/novice/cv"))
OUT_DIR = Path(os.getenv("CV_PREPARED_DIR", "/home/jupyter/BrainHack_V2/cv/data/til26"))


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


def prepare_dataset() -> None:
    images_dir = NOVICE_CV_DIR / "images"
    ann_path = NOVICE_CV_DIR / "annotations.json"
    
    if not ann_path.exists():
        print(f"Error: {ann_path} not found!")
        return

    with open(ann_path, "r", encoding="utf-8") as f:
        annotations_json = json.load(f)
        
    images = annotations_json["images"]
    annotations = annotations_json["annotations"]
    categories = annotations_json["categories"]

    category_ids = sorted({int(annotation["category_id"]) for annotation in annotations})
    cat_to_yolo = {category_id: index for index, category_id in enumerate(category_ids)}

    anns_by_image = {}
    for annotation in annotations:
        anns_by_image.setdefault(annotation["image_id"], []).append(annotation)

    random.seed(26)
    shuffled = list(images)
    random.shuffle(shuffled)
    cut = int(len(shuffled) * 0.9)
    splits = {"train": shuffled[:cut], "val": shuffled[cut:]}

    # 1. Create YOLO structure
    for subdir in ("images/train", "images/val", "labels/train", "labels/val"):
        target = OUT_DIR / subdir
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    # 2. Create COCO structure (RF-DETR)
    for subdir in ("train", "valid"):
        target = OUT_DIR / subdir
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    # 3. Process splits
    for split, split_images in splits.items():
        coco_split_name = "train" if split == "train" else "valid"
        coco_images = []
        coco_annotations = []
        
        for image in split_images:
            name = str(image["file_name"])
            src_img = images_dir / name
            if not src_img.exists():
                continue
            
            # YOLO images
            link_or_copy(src_img, OUT_DIR / f"images/{split}" / name)
            # COCO images
            link_or_copy(src_img, OUT_DIR / coco_split_name / name)

            coco_images.append(image)
            coco_annotations.extend(anns_by_image.get(image["id"], []))

            # YOLO labels
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

        # Write split COCO annotations
        coco_json = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories
        }
        coco_ann_file = OUT_DIR / coco_split_name / "_annotations.coco.json"
        with open(coco_ann_file, "w", encoding="utf-8") as f:
            json.dump(coco_json, f, ensure_ascii=False)

    # Symlink val to valid
    val_sym = OUT_DIR / "val"
    if val_sym.exists() or val_sym.is_symlink():
        val_sym.unlink()
    try:
        os.symlink("valid", val_sym, target_is_directory=True)
    except OSError:
        pass

    print(f"Successfully prepared dual-format YOLO and COCO dataset!")
    print(f"Train images: {len(splits['train'])}")
    print(f"Val images: {len(splits['val'])}")
    print(f"Output saved to: {OUT_DIR}")


if __name__ == "__main__":
    prepare_dataset()
