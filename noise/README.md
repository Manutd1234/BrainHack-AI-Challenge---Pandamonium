# Noise: YOLO-Surrogate Adversarial Perturbation

This module serves the Noise challenge on `POST /noise` at port `5003`.

In Finals, Noise can be used to perturb an opponent's CV input image while remaining visually similar enough to pass the evaluator's image-similarity constraints. The objective is indirect: make the opponent's detector less reliable without violating similarity checks.

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
    "BASE64_ENCODED_NOISED_JPEG"
  ]
}
```

## Architecture

The current manager implements:

- JPEG decode and encode.
- Optional resize for attack efficiency.
- YOLO surrogate model loaded from `model/best.pt`.
- PGD-style adversarial perturbation against the surrogate detector.
- L-infinity perturbation clipping.
- SSIM-guided projection / bisection back toward the original image.
- Deterministic fallback noise if no surrogate is available.

## Why YOLO Surrogate

The opponent CV systems are likely YOLO-like because the challenge target set is object detection with bounding boxes and class IDs. Attacking a local YOLO surrogate is a practical transfer attack: perturbations that reduce surrogate confidence may also disrupt other detectors trained on similar data.

## Runtime Environment Variables

| Variable | Purpose |
| --- | --- |
| `CV_MODEL_PATH` | surrogate YOLO checkpoint |
| `NOISE_EPSILON` | L-infinity perturbation budget |
| `NOISE_ALPHA` | PGD step size |
| `NOISE_PGD_STEPS` | number of PGD updates |
| `NOISE_SSIM_MIN` | minimum SSIM target |
| `NOISE_BISECT_ITERS` | projection iterations |
| `NOISE_JPEG_QUALITY` | output JPEG quality |
| `NOISE_MAX_SIDE` | resize limit for attack compute |

## Setup

Use the trained CV checkpoint as the surrogate:

```bash
cd /home/jupyter/BrainHack_clean/BrainHack_V2/noise
mkdir -p model
cp ../cv/model/best.pt model/best.pt
```

## Build and Submit

```bash
export TIL_FOLDER=/home/jupyter/BrainHack_clean/BrainHack_V2
cd "$TIL_FOLDER/noise"

til build noise v1
til test noise v1
til submit noise v1
```

## Tuning Notes

- Increase `NOISE_PGD_STEPS` for stronger attacks, but expect slower inference.
- Increase `NOISE_EPSILON` carefully; too high may fail similarity checks.
- Lower `NOISE_SSIM_MIN` only if the finals evaluator permits it.
- Keep a deterministic fallback so malformed images do not crash the service.
