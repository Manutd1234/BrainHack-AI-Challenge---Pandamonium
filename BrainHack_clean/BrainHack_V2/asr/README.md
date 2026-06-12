# ASR

Your ASR challenge is to transcribe a noisy recording of speech.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/asr` route on port `5001`. It is a JSON document structured as such:

```JSON
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_AUDIO"
    },
    ...
  ]
}
```

The `b64` key of each object in the `instances` list contains the base64-encoded bytes of the input audio in WAV format. The length of the `instances` list is variable.

## Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        "Predicted transcript one.",
        "Predicted transcript two.",
        ...
    ]
}
```

where each string in `predictions` is the predicted ASR transcription for the corresponding audio file.

The $k$-th element of `predictions` must be the prediction corresponding to the $k$-th element of `instances` for all $1 \le k \le n$, where n is the number of input instances. The length of `predictions` must equal that of `instances`.

## Implementation

This container uses MERaLiON-2-3B for transcription and DeepFilterNet3 for
speech enhancement before inference. The Docker build downloads the MERaLiON
weights into the image with `huggingface_hub` and warms the DeepFilterNet3 cache
so the runtime container can serve without network access.

Useful environment variables:

- `MERALION_MODEL_PATH`: local model path baked into the image.
- `ASR_USE_DEEPFILTERNET`: set to `0` to disable DeepFilterNet3.
- `ASR_MAX_SECONDS`: max audio duration sent to MERaLiON, default `30`.
- `ASR_MAX_NEW_TOKENS`: generation budget, default `128`.

Build from this directory:

```bash
docker build -t pandamonium-asr:v2 .
```

Run:

```bash
docker run --gpus all -p 5001:5001 pandamonium-asr:v2
```
