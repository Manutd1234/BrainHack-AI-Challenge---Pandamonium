# CV: YOLOv11 Object Detection

This module serves the CV challenge on `POST /cv` at port `5002`.

The task is to detect and classify challenge objects in JPEG images and return boxes in LTWH format: `[left, top, width, height]`. The official scoring uses mAP over IoU thresholds from 0.50 to 0.95.

## Target Classes

The challenge target list has 18 classes:

```text
0 cargo aircraft
1 commercial aircraft
2 drone
3 fighter jet
4 fighter plane
5 helicopter
6 light aircraft
7 missile
8 truck
9 car
10 tank
11 bus
12 van
13 cargo ship
14 yacht
15 cruise ship
16 warship
17 sailboat
```

## Input and Output

Input:

```json
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_JPEG"
    }
  ]
}
```

Output:

```json
{
  "predictions": [
    [
      {
        "bbox": [10, 20, 100, 80],
        "category_id": 2
      }
    ]
  ]
}
```

## Architecture

The manager loads one or more Ultralytics YOLO checkpoints and returns TIL-format detections.

Key runtime features:

- YOLOv11/Ultralytics checkpoint loading.
- Optional model ensemble through `CV_MODEL_PATHS`.
- Per-class confidence thresholds.
- Full-image inference as the main path.
- Optional high-resolution fallback for sparse detections.
- Optional SAHI slicing for large images.
- Weighted box fusion / final NMS style postprocessing.
- Conversion from XYXY/XYWH-style model boxes to LTWH challenge format.

## Training

Train on the challenge data using the provided scripts:

```bash
cd /home/jupyter/BrainHack_clean/BrainHack_V2/cv
python cv_train.py
```

Refinement scripts:

```bash
python cv_train_final.py
python cv_refine_train.py
python tune_thresholds.py
```

The expected inference checkpoint is:

```text
cv/model/best.pt
```

## Threshold Tuning

`src/cv_thresholds.json` stores tuned confidence and IoU settings. Tuning is important because the classes have different false-positive/false-negative costs:

- low threshold for small/faint objects such as drones and missiles,
- higher threshold for large vehicle/vessel classes,
- final NMS to merge overlapping predictions,
- optional high-resolution fallback when normal inference returns too few boxes.

## Runtime Environment Variables

| Variable | Purpose |
| --- | --- |
| `CV_MODEL_PATH` | default checkpoint path |
| `CV_MODEL_PATHS` | comma-separated ensemble checkpoint paths |
| `CV_DEFAULT_CONF` | default confidence threshold |
| `CV_IOU` | model NMS IoU |
| `CV_IMGSZ` | inference image size |
| `CV_MAX_DETECTIONS` | final maximum returned detections |
| `CV_USE_SAHI` | enable sliced inference |
| `CV_HIGHRES_FALLBACK` | enable high-resolution fallback |
| `CV_USE_WBF` | enable weighted box fusion style merging |

## Build and Submit

```bash
export TIL_FOLDER=/home/jupyter/BrainHack_clean/BrainHack_V2
cd "$TIL_FOLDER/cv"

ls -lh model/best.pt
til build cv v1
til test cv v1
til submit cv v1
```

## Debug Checklist

- If local test returns boxes but score is low, confirm LTWH format.
- If all predictions are empty, check `model/best.pt` and confidence thresholds.
- If many duplicate boxes appear, tune final NMS/WBF settings.
- If small targets are missed, test higher `CV_IMGSZ` or SAHI.
