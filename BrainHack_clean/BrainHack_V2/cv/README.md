# CV

Your CV challenge is to detect and classify objects in an image.

## Input

The input is sent via a POST request to the `/cv` route on port `5002`. It is a
JSON document structured as such:

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

The `b64` key contains the base64-encoded JPEG bytes for each input image.

## Output

The route returns:

```Python
{
    "predictions": [
        [
            {
                "bbox": [x, y, w, h],
                "category_id": category_id
            }
        ]
    ]
}
```

The bounding box format is left/top/width/height (`[l, t, w, h]`), not YOLO's
center/width/height format.

## Implementation

This container uses YOLO26x with SAHI sliced inference for larger images. SAHI's
NMS postprocess is used to merge slice predictions while the full-image fallback
also keeps the same `[l, t, w, h]` response contract.

Train first on a GPU machine from this directory:

```bash
python cv_train.py
```

That writes `model/best.pt`. Build only after the checkpoint exists:

```bash
docker build -t pandamonium-cv:v2 .
docker run --gpus all -p 5002:5002 pandamonium-cv:v2
```
