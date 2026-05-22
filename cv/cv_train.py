"""
CV Training — trains RF-DETR-large and/or YOLO26x on aerial dataset.
Run: python cv_train.py --model rfdetr
     python cv_train.py --model yolo
     python cv_train.py --model both
"""
import argparse
import os
import shutil
import yaml

CLASSES = [
    "cargo aircraft", "commercial aircraft", "drone", "fighter jet", "fighter plane",
    "helicopter", "light aircraft", "missile", "truck", "car", "tank", "bus", "van",
    "cargo ship", "yacht", "cruise ship", "warship", "sailboat"
]

def train_rfdetr(data_dir="data/til26"):
    print("=" * 60)
    print("STARTING RF-DETR-LARGE TRAINING (50 EPOCHS)")
    print("=" * 60)
    from rfdetr import RFDETRLarge
    model = RFDETRLarge(num_classes=len(CLASSES), pretrained=True)
    model.train(
        dataset_dir=data_dir, epochs=50, batch_size=8,
        lr=1e-4, lr_encoder=1e-5, resolution=800,
        grad_accumulation_steps=2, num_workers=4,
        output_dir="runs/rfdetr_til26",
    )
    os.makedirs("model", exist_ok=True)
    shutil.copy("runs/rfdetr_til26/checkpoint_best.pth", "model/rfdetr_best.pt")
    print("RF-DETR training complete → model/rfdetr_best.pt")

def train_yolo(data_yaml="data/til26.yaml"):
    print("=" * 60)
    print("STARTING YOLO26X TRAINING (50 EPOCHS)")
    print("=" * 60)
    from ultralytics import YOLO
    if not os.path.exists(data_yaml):
        cfg = {
            "path": os.path.abspath("data/til26"),
            "train": "images/train",
            "val": "images/val",
            "names": {i: n for i, n in enumerate(CLASSES)},
            "nc": len(CLASSES)
        }
        with open(data_yaml, "w") as f:
            yaml.dump(cfg, f, allow_unicode=True)
            
    model = YOLO("yolo26x.pt")
    model.train(
        data=data_yaml, epochs=50, imgsz=1280, batch=4, device="cuda",
        degrees=90.0, flipud=0.5, fliplr=0.5, mosaic=1.0, mixup=0.15,
        copy_paste=0.3, erasing=0.4, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        optimizer="auto", lr0=0.01, lrf=0.01, warmup_epochs=3,
        project="runs", name="yolo26x_til26", exist_ok=True,
        save=True, val=True, plots=True,
    )
    os.makedirs("model", exist_ok=True)
    shutil.copy("runs/yolo26x_til26/weights/best.pt", "model/yolo_best.pt")
    shutil.copy("runs/yolo26x_til26/weights/best.pt", "model/best.pt")
    print("YOLO26x training complete → model/best.pt & model/yolo_best.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rfdetr", "yolo", "both"], default="both")
    args = p.parse_args()
    if args.model in ("rfdetr", "both"):
        train_rfdetr()
    if args.model in ("yolo", "both"):
        train_yolo()
