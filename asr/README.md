# ASR: Parakeet Fine-Tuned Speech Recognition

This module serves the Automatic Speech Recognition challenge on `POST /asr` at port `5001`.

The Novice ASR track is English speech with accents and fictional-world terms from the TIL-AI corpus. The scoring transform lowercases text, removes punctuation, normalizes hyphens, strips whitespace, and computes `max(0, 1 - WER)` for English.

## Input and Output

Input:

```json
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_WAV"
    }
  ]
}
```

Output:

```json
{
  "predictions": [
    "Predicted transcript"
  ]
}
```

The output list length must match the input list length.

## Current Model Strategy

The ASR stack is built around NVIDIA Parakeet TDT:

- Base family: `parakeet-tdt-0.6b-v2` and `parakeet-tdt-0.6b-v3`.
- Best local checkpoint family: decoder fine-tuned Parakeet v2.
- Strongest checkpoint in experiments: `parakeet_v2_decoder_e1_hard120_replay.nemo`.
- Inference: FP16 where possible, TF32 enabled, CUDA graph decoding disabled for TDT/RNNT safety.
- Batching: length-sorted batches to reduce padding.
- Runtime cap: tuned between 33 and 36 seconds depending on speed/accuracy target.
- Postprocessing: safe phrase corrections only; broad vocabulary forcing, global number expansion, time-stretch compression, and aggressive VAD were tested and avoided when harmful.

## Why Parakeet

Parakeet TDT performs well on single-speaker English speech, handles accents robustly, and runs quickly enough on the challenge GPU. The model is a better fit than general Whisper variants for this finals workload because the input format is short WAV audio and the scoring cares about word-level accuracy and speed.

## Fine-Tuning Approach

The successful training recipe was conservative:

1. Start from a Parakeet v2 `.nemo` checkpoint.
2. Fine-tune decoder or final layers only.
3. Use low learning rates to avoid damaging the acoustic encoder.
4. Mine high-WER examples for a small hard replay set.
5. Evaluate every candidate on the same first500 benchmark.
6. Reject additional training when it reduces first500 or full-eval generalization.

Training command pattern:

```bash
cd /home/jupyter/asr/finetune_v3

export CUDA_VISIBLE_DEVICES=0
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python finetune_parakeet_v3.py \
  --base /home/jupyter/asr/model/parakeet/parakeet-tdt-0.6b-v2.nemo \
  --train /home/jupyter/asr/finetune_v3/train_manifest.jsonl \
  --val /home/jupyter/asr/finetune_v3/val_manifest.jsonl \
  --out /home/jupyter/asr/finetune_v3/parakeet_v2_candidate.nemo \
  --phase decoder \
  --epochs 1 \
  --batch 2 \
  --lr 0.000005 \
  --workers 2
```

Important finding: repeated fine-tuning on the same hard examples can overfit and reduce hidden score. The best candidate was not the most trained candidate.

## Correction Pipeline

`src/asr_manager.py` loads `asr_corrections.json` and applies phrase-level corrections. Good corrections are domain-specific and low-risk:

- `Cyanite`, `Phyrexis`, `Nyari`, `Kashikari`, `Sarento`, `Renhwa`
- `CYPHER`, `TEC`, `CGC`, `WTO`
- `Cape Tidak`, `New Mewan`, `Soo Hyun`, `Veyanova`

Avoid broad corrections:

- common words such as `the`, `and`, `one`, `five`
- capitalization-only fixes
- style variants such as `All right` versus `Alright`
- US/UK spelling swaps such as `gray`/`grey`, `labor`/`labour`

## Runtime Environment Variables

| Variable | Purpose |
| --- | --- |
| `ASR_MODEL_PATH` | `.nemo` checkpoint path inside or outside the container |
| `ASR_CORRECTIONS_FILE` | phrase correction JSON |
| `ASR_VOCABULARY_FILE` | optional vocabulary JSON |
| `ASR_USE_MEMORY` | disables or enables exact audio-memory lookup |
| `ASR_MAX_SECONDS` | audio duration cap before inference |
| `ASR_BATCH_SIZE` | Parakeet transcribe batch size |
| `ASR_FP16` | enable half precision on CUDA |
| `ASR_DISABLE_CUDA_GRAPHS` | disable dynamic RNNT/TDT graph decoding |

## Local Evaluation

```bash
cd /home/jupyter/asr/src

ASR_MODEL_PATH=/home/jupyter/asr/finetune_v3/parakeet_v2_decoder_e1_hard120_replay.nemo \
ASR_CORRECTIONS_FILE=/home/jupyter/asr/src/asr_corrections.json \
ASR_USE_MEMORY=0 \
ASR_BATCH_SIZE=8 \
ASR_MAX_SECONDS=34 \
ASR_FP16=1 \
ASR_DISABLE_CUDA_GRAPHS=1 \
python eval_asr.py \
  --limit 500 \
  --batch 8 \
  --out /home/jupyter/asr/runs/eval_candidate_first500

cat /home/jupyter/asr/runs/eval_candidate_first500/eval_summary.json
```

## Build and Submit

Place the selected model in `asr/model/parakeet/`, then build:

```bash
export TIL_FOLDER=/home/jupyter/BrainHack_clean/BrainHack_V2
cd "$TIL_FOLDER/asr"

til build asr v43
til test asr v43
til submit asr v43
```

## Lessons Learned

- `30s` is very fast but clips too many endings.
- `33s` to `34s` is the best hidden speed/accuracy tradeoff observed.
- Time-stretching long clips degraded both accuracy and latency.
- Conservative silence trimming was not a reliable win.
- Full vocabulary forcing reduced exact match and should remain off unless retested.
- CUDA graphs can fail or behave inconsistently with dynamic TDT/RNNT decoding, so the stable build disables them.
