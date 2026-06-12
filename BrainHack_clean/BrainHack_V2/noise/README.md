# Noise

Your Noise challenge is to adversarially perturb JPEG images while keeping them
visually similar to the input.

## Input

The input is sent via `POST /noise` on port `5003`:

```JSON
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_IMAGE"
    }
  ]
}
```

## Output

The response is:

```JSON
{
  "predictions": [
    "BASE64_ENCODED_NOISED_IMAGE"
  ]
}
```

## Implementation

This container performs a PGD white-box attack against a surrogate YOLO CV model
loaded from `model/best.pt`. The perturbation uses an `8/255` L-infinity budget
by default and then bisects the perturbation scale until the output satisfies
the configured SSIM floor, default `0.85`.

Useful environment variables:

- `NOISE_EPSILON`: PGD L-infinity budget, default `8/255`.
- `NOISE_ALPHA`: PGD step size, default `2/255`.
- `NOISE_PGD_STEPS`: PGD steps, default `20`.
- `NOISE_SSIM_MIN`: minimum SSIM, default `0.85`.
- `NOISE_JPEG_QUALITY`: JPEG output quality, default `95`.

Before building, copy the trained CV checkpoint into this module:

```bash
cp ../cv/model/best.pt model/best.pt
```

Build and run:

```bash
docker build -t pandamonium-noise:v2 .
docker run --gpus all -p 5003:5003 pandamonium-noise:v2
```
