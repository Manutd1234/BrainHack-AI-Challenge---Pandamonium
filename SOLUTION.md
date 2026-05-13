# BrainHack TIL-26 Pandamonium Solution

This repository is wired for the high-capacity 2026 stack:

| Track | Runtime strategy | Primary files |
| --- | --- | --- |
| NLP | BGE-M3 dense retrieval + BM25 reciprocal-rank fusion + Qwen3.6 via vLLM tensor parallelism | `nlp/src/nlp_manager.py`, `nlp/start_server.sh` |
| CV | YOLO26 with SAHI-style sliced inference, 30% overlap, TIL-26 class mapping fallback | `cv/src/cv_manager.py` |
| Noise | PGD against the same YOLO26 family used by CV, with bounded texture/color perturbations | `noise/src/noise_manager.py` |
| AE | Discrete SAC with ResNet bottleneck encoders for `agent_viewcone` and `base_viewcone` | `ae/train.py`, `ae/src/ae_model.py` |
| ASR | faster-whisper large-v3 FP16, optional bundled fine-tuned CTranslate2 model | `asr/src/asr_manager.py` |

## GPU Allocation

Recommended qualifier allocation:

- NLP: 2 GPUs, `QWEN_TP_SIZE=2`
- CV + Noise + ASR: 1 shared GPU if orchestrated together, or one GPU per submitted container during isolated tests
- AE: 1 GPU for training; inference can run on CPU or GPU

## NLP

The NLP Docker image starts vLLM automatically by default:

```bash
ENABLE_LOCAL_VLLM=1
QWEN_MODEL=Qwen/Qwen3.6-27B
QWEN_TP_SIZE=2
QWEN_MAX_MODEL_LEN=131072
```

The Qwen weights are intentionally not committed. For a final offline build,
bundle them into the image:

```bash
cd nlp
docker build --build-arg DOWNLOAD_QWEN=1 -t til-nlp-qwen36 .
```

For lightweight smoke tests without Qwen:

```bash
NLP_USE_LLM=0 python download_models.py --skip-qwen
```

## CV And Noise

Place fine-tuned TIL-26 weights at:

```text
cv/models/best.pt
noise/models/best.pt
```

If no fine-tuned checkpoint exists, both tracks fall back to `yolo26l.pt`.

CV knobs:

```bash
YOLO_IMGSZ=1280
SAHI_SLICE_SIZE=1024
SAHI_OVERLAP=0.30
YOLO_CONF=0.12
```

Noise knobs:

```bash
NOISE_EPSILON=10.0
NOISE_PGD_STEPS=8
NOISE_SURROGATE_IMGSZ=640
```

## AE

Train the discrete SAC policy on a GPU workbench:

```bash
cd ae
python train.py --mode advanced --envs 8 --total-steps 5000000
```

The trained inference checkpoint is saved to:

```text
ae/models/ae/sac_resnet_policy.pt
```

The AE server falls back to a deterministic exploration heuristic when no SAC
checkpoint is present, so submissions remain functional during iteration.

## ASR

By default ASR uses `large-v3` in faster-whisper. To ship a fine-tuned model,
place the CTranslate2 export at:

```text
asr/models/whisper-large-v3-finetuned
```

Useful runtime knobs:

```bash
WHISPER_MODEL=models/whisper-large-v3-finetuned
WHISPER_FALLBACK_MODEL=large-v3
WHISPER_DEVICE_INDEX=0
```
