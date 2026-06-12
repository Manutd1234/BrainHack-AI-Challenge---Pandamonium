"""Decoder-only fine-tuning for NVIDIA Parakeet-TDT 0.6B on novice ASR data.

This follows the high-speed recipe that worked well for other teams:

1. Start from nvidia/parakeet-tdt-0.6b-v2.
2. Freeze the encoder and preprocessor.
3. Fine-tune decoder/joint-style layers only.
4. Use mixed precision on GPU.
5. Save a .nemo file into ./model/parakeet so the Dockerfile bakes it in.

Run on the Jupyter machine:

    cd ~/asr
    pip install -r requirements-train.txt
    ASR_TRAIN_STEPS=1500 ASR_TRAIN_BATCH=4 python train_parakeet_decoder.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import soundfile as sf
import torch


DATA_DIR = Path(os.getenv("ASR_DATA_DIR", "/home/jupyter/novice/asr"))
MODEL_NAME = os.getenv("ASR_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")
MODEL_SLUG = MODEL_NAME.split("/")[-1]
RUN_DIR = Path(os.getenv("ASR_RUN_DIR", "/home/jupyter/asr/runs/parakeet_decoder"))
OUTPUT_NEMO = Path(
    os.getenv(
        "ASR_OUTPUT_NEMO",
        f"/home/jupyter/asr/model/parakeet/{MODEL_SLUG}.nemo",
    )
)
SEED = int(os.getenv("ASR_SEED", "26"))
VAL_FRACTION = float(os.getenv("ASR_VAL_FRACTION", "0.05"))
MAX_STEPS = int(os.getenv("ASR_TRAIN_STEPS", "1500"))
BATCH_SIZE = int(os.getenv("ASR_TRAIN_BATCH", "4"))
GRAD_ACCUM = int(os.getenv("ASR_GRAD_ACCUM", "4"))
LEARNING_RATE = float(os.getenv("ASR_LR", "3e-5"))
NUM_WORKERS = int(os.getenv("ASR_NUM_WORKERS", "2"))


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _audio_field(item: dict[str, Any]) -> str:
    for key in ("audio", "file", "path", "wav"):
        if item.get(key):
            return str(item[key])
    raise KeyError(f"No audio path field in item: {item.keys()}")


def _transcript_field(item: dict[str, Any]) -> str:
    for key in ("transcript", "text", "sentence"):
        if item.get(key):
            return str(item[key]).strip()
    return ""


def _duration_seconds(audio_path: Path) -> float:
    info = sf.info(str(audio_path))
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def _read_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest = DATA_DIR / "asr.jsonl"
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            text = _transcript_field(item)
            if not text:
                continue
            audio_path = DATA_DIR / _audio_field(item)
            if not audio_path.exists():
                continue
            rows.append(
                {
                    "audio_filepath": str(audio_path),
                    "text": text,
                    "duration": _duration_seconds(audio_path),
                }
            )
    if not rows:
        raise RuntimeError(f"No ASR training rows found in {manifest}")
    return rows


def _write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_manifests() -> tuple[Path, Path]:
    rows = _read_rows()
    random.Random(SEED).shuffle(rows)
    val_size = max(1, int(len(rows) * VAL_FRACTION))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    train_manifest = RUN_DIR / "train_manifest.json"
    val_manifest = RUN_DIR / "val_manifest.json"
    _write_manifest(train_rows, train_manifest)
    _write_manifest(val_rows, val_manifest)
    print(f"Train rows: {len(train_rows)}")
    print(f"Validation rows: {len(val_rows)}")
    return train_manifest, val_manifest


def _load_lightning():
    try:
        import lightning.pytorch as pl

        return pl
    except ImportError:
        import pytorch_lightning as pl

        return pl


def _load_model():
    import nemo.collections.asr as nemo_asr

    pretrained_nemo = Path(os.getenv("ASR_PRETRAINED_NEMO", ""))
    if pretrained_nemo.exists():
        print(f"Restoring {pretrained_nemo}")
        return nemo_asr.models.ASRModel.restore_from(str(pretrained_nemo))
    print(f"Loading {MODEL_NAME}")
    return nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)


def _freeze_for_decoder_only(model) -> int:
    trainable_keywords = tuple(
        token.strip().lower()
        for token in os.getenv(
            "ASR_TRAINABLE_KEYWORDS",
            "decoder,joint,predictor,transf_decoder,tdt",
        ).split(",")
        if token.strip()
    )
    frozen_keywords = tuple(
        token.strip().lower()
        for token in os.getenv(
            "ASR_FROZEN_KEYWORDS",
            "encoder,preprocessor,frontend,feature",
        ).split(",")
        if token.strip()
    )

    trainable_count = 0
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        should_train = any(token in lowered for token in trainable_keywords)
        should_freeze = any(token in lowered for token in frozen_keywords)
        parameter.requires_grad = should_train and not should_freeze
        if parameter.requires_grad:
            trainable_count += parameter.numel()

    if trainable_count == 0:
        raise RuntimeError(
            "No decoder parameters were left trainable. Inspect model.named_parameters() "
            "and adjust ASR_TRAINABLE_KEYWORDS."
        )
    print(f"Trainable decoder/joint parameters: {trainable_count:,}")
    return trainable_count


def _data_config(manifest_path: Path, shuffle: bool) -> dict[str, Any]:
    return {
        "manifest_filepath": str(manifest_path),
        "sample_rate": 16000,
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "max_duration": float(os.getenv("ASR_MAX_DURATION", "35.0")),
        "min_duration": float(os.getenv("ASR_MIN_DURATION", "0.1")),
    }


def _set_optimizer(model) -> None:
    from omegaconf import OmegaConf, open_dict

    optim = OmegaConf.create(
        {
            "name": "adamw",
            "lr": LEARNING_RATE,
            "betas": [0.9, 0.98],
            "weight_decay": float(os.getenv("ASR_WEIGHT_DECAY", "0.001")),
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": int(os.getenv("ASR_WARMUP_STEPS", "100")),
                "min_lr": float(os.getenv("ASR_MIN_LR", "1e-6")),
            },
        }
    )
    with open_dict(model.cfg):
        model.cfg.optim = optim


def train() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NEMO.parent.mkdir(parents=True, exist_ok=True)
    train_manifest, val_manifest = _prepare_manifests()

    pl = _load_lightning()
    pl.seed_everything(SEED, workers=True)

    model = _load_model()
    model.setup_training_data(_data_config(train_manifest, shuffle=True))
    model.setup_validation_data(_data_config(val_manifest, shuffle=False))
    _freeze_for_decoder_only(model)
    _set_optimizer(model)

    trainer_kwargs = {
        "default_root_dir": str(RUN_DIR),
        "max_steps": MAX_STEPS,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "accumulate_grad_batches": GRAD_ACCUM,
        "log_every_n_steps": int(os.getenv("ASR_LOG_STEPS", "25")),
        "val_check_interval": int(os.getenv("ASR_EVAL_STEPS", "250")),
        "enable_checkpointing": True,
        "gradient_clip_val": float(os.getenv("ASR_GRAD_CLIP", "1.0")),
    }
    if torch.cuda.is_available() and _env_flag("ASR_MIXED_PRECISION", True):
        trainer_kwargs["precision"] = "16-mixed"

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model)

    if OUTPUT_NEMO.exists() and _env_flag("ASR_BACKUP_EXISTING_NEMO", True):
        backup = OUTPUT_NEMO.with_suffix(".before_decoder_ft.nemo")
        if not backup.exists():
            OUTPUT_NEMO.replace(backup)
            print(f"Backed up previous checkpoint to {backup}")

    model.save_to(str(OUTPUT_NEMO))
    print(f"Saved decoder-tuned Parakeet checkpoint to {OUTPUT_NEMO}")


if __name__ == "__main__":
    train()
