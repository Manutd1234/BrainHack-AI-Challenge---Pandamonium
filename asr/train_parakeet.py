"""Fine-tune nvidia/parakeet-tdt-0.6b-v3 on novice ASR data.

Differences vs. the previous decoder-only script:

  * Full fine-tuning by default (set ASR_DECODER_ONLY=1 to fall back).
  * SpecAugment ON during training (mask freq + time bins).
  * Optional 3x speed perturbation on the training manifest.
  * Differential learning rates: encoder gets LR * 0.1, decoder/joint gets LR.
  * WER validation every N steps with a `val_loss` + `val_wer` log.
  * Saves both a `.nemo` and the best-WER checkpoint.

Run:
    cd ~/asr
    pip install -r requirements-train.txt
    ASR_TRAIN_STEPS=4000 ASR_TRAIN_BATCH=8 python train_parakeet.py
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
MODEL_NAME = os.getenv("ASR_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")
MODEL_SLUG = MODEL_NAME.split("/")[-1]
RUN_DIR = Path(os.getenv("ASR_RUN_DIR", "/home/jupyter/asr/runs/parakeet_full_ft"))
OUTPUT_NEMO = Path(
    os.getenv(
        "ASR_OUTPUT_NEMO",
        f"/home/jupyter/asr/model/parakeet/{MODEL_SLUG}.nemo",
    )
)

SEED = int(os.getenv("ASR_SEED", "26"))
VAL_FRACTION = float(os.getenv("ASR_VAL_FRACTION", "0.04"))
MAX_STEPS = int(os.getenv("ASR_TRAIN_STEPS", "4000"))
BATCH_SIZE = int(os.getenv("ASR_TRAIN_BATCH", "8"))
GRAD_ACCUM = int(os.getenv("ASR_GRAD_ACCUM", "2"))
ENCODER_LR = float(os.getenv("ASR_ENCODER_LR", "3e-6"))
DECODER_LR = float(os.getenv("ASR_DECODER_LR", "3e-5"))
NUM_WORKERS = int(os.getenv("ASR_NUM_WORKERS", "4"))
DECODER_ONLY = os.getenv("ASR_DECODER_ONLY", "0").strip().lower() in {"1", "true", "yes"}
USE_SPEED_PERT = os.getenv("ASR_SPEED_PERT", "1").strip().lower() in {"1", "true", "yes"}


def _flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------- manifest building ----------

def _audio_field(item: dict[str, Any]) -> str:
    for key in ("audio", "file", "path", "wav"):
        if item.get(key):
            return str(item[key])
    raise KeyError(f"No audio path field in item: {list(item.keys())}")


def _transcript_field(item: dict[str, Any]) -> str:
    for key in ("transcript", "text", "sentence"):
        if item.get(key):
            return str(item[key]).strip()
    return ""


def _duration_s(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate) if info.samplerate else 0.0


def _read_rows() -> list[dict[str, Any]]:
    manifest = DATA_DIR / "asr.jsonl"
    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            text = _transcript_field(item)
            if not text:
                continue
            audio = DATA_DIR / _audio_field(item)
            if not audio.exists():
                continue
            rows.append({"audio_filepath": str(audio), "text": text, "duration": _duration_s(audio)})
    if not rows:
        raise RuntimeError(f"No training rows found in {manifest}")
    return rows


def _write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _prepare_manifests() -> tuple[Path, Path]:
    rows = _read_rows()
    random.Random(SEED).shuffle(rows)
    val_size = max(1, int(len(rows) * VAL_FRACTION))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    train_path = RUN_DIR / "train_manifest.json"
    val_path = RUN_DIR / "val_manifest.json"
    _write_manifest(train_rows, train_path)
    _write_manifest(val_rows, val_path)
    print(f"train rows={len(train_rows)} val rows={len(val_rows)}")
    return train_path, val_path


# ---------- model setup ----------

def _load_lightning():
    try:
        import pytorch_lightning as pl
        return pl
    except ImportError:
        import lightning.pytorch as pl
        return pl


def _load_model():
    import nemo.collections.asr as nemo_asr
    pre = Path(os.getenv("ASR_PRETRAINED_NEMO", ""))
    if pre.exists():
        print(f"restoring {pre}")
        return nemo_asr.models.ASRModel.restore_from(str(pre))
    print(f"loading {MODEL_NAME}")
    return nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)


def _configure_freeze(model) -> None:
    if not DECODER_ONLY:
        # Everything trainable. Encoder LR is reduced via optimizer param groups.
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"full fine-tune: {trainable:,} params")
        return

    trainable_kw = ("decoder", "joint", "predictor", "transf_decoder", "tdt")
    frozen_kw = ("encoder", "preprocessor", "frontend", "feature")
    count = 0
    for name, p in model.named_parameters():
        n = name.lower()
        p.requires_grad = any(k in n for k in trainable_kw) and not any(k in n for k in frozen_kw)
        if p.requires_grad:
            count += p.numel()
    if count == 0:
        raise RuntimeError("decoder-only freeze left zero trainable params; check keyword lists")
    print(f"decoder-only fine-tune: {count:,} params")


def _enable_spec_augment(model) -> None:
    """Turn on SpecAugment via the preprocessor or a model-level spec_augment cfg."""
    from omegaconf import OmegaConf, open_dict

    spec = OmegaConf.create({
        "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
        "freq_masks": int(os.getenv("ASR_FREQ_MASKS", "2")),
        "time_masks": int(os.getenv("ASR_TIME_MASKS", "10")),
        "freq_width": int(os.getenv("ASR_FREQ_WIDTH", "27")),
        "time_width": float(os.getenv("ASR_TIME_WIDTH", "0.05")),
    })
    with open_dict(model.cfg):
        model.cfg.spec_augment = spec
    # Re-instantiate the augmentor on the live model if it exposes one.
    if hasattr(model, "spec_augmentation"):
        try:
            from nemo.collections.asr.modules import SpectrogramAugmentation
            model.spec_augmentation = SpectrogramAugmentation(
                freq_masks=spec.freq_masks,
                time_masks=spec.time_masks,
                freq_width=spec.freq_width,
                time_width=spec.time_width,
            )
            print("SpecAugment enabled")
        except Exception as exc:
            print(f"warn: could not re-instantiate spec_augmentation: {exc}")


def _data_cfg(manifest: Path, shuffle: bool, training: bool) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "manifest_filepath": str(manifest),
        "sample_rate": 16000,
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "max_duration": float(os.getenv("ASR_MAX_DURATION", "35.0")),
        "min_duration": float(os.getenv("ASR_MIN_DURATION", "0.3")),
    }
    if training and USE_SPEED_PERT:
        # NeMo applies these on-the-fly if the loader supports it.
        cfg["augmentor"] = {
            "speed": {"prob": 0.5, "sr": 16000, "resample_type": "kaiser_fast",
                       "min_speed_rate": 0.95, "max_speed_rate": 1.05, "num_rates": 3},
        }
    return cfg


def _set_optimizer(model) -> None:
    """Two param groups: encoder (low LR) vs everything else (full LR)."""
    from omegaconf import OmegaConf, open_dict

    # Even if we use the OmegaConf optimizer NeMo will build one group; we
    # patch the actual param groups in a configure_optimizers hook below.
    optim = OmegaConf.create({
        "name": "adamw",
        "lr": DECODER_LR,
        "betas": [0.9, 0.98],
        "weight_decay": float(os.getenv("ASR_WEIGHT_DECAY", "1e-3")),
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": int(os.getenv("ASR_WARMUP_STEPS", "200")),
            "min_lr": float(os.getenv("ASR_MIN_LR", "1e-6")),
        },
    })
    with open_dict(model.cfg):
        model.cfg.optim = optim

    if DECODER_ONLY:
        return

    # Override configure_optimizers to apply differential LRs.
    original_configure = model.configure_optimizers

    def configure_optimizers():  # type: ignore[no-redef]
        encoder_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (encoder_params if name.lower().startswith(("encoder.", "preprocessor."))
             else other_params).append(p)
        opt = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": ENCODER_LR},
                {"params": other_params, "lr": DECODER_LR},
            ],
            betas=(0.9, 0.98),
            weight_decay=float(os.getenv("ASR_WEIGHT_DECAY", "1e-3")),
        )
        # Reuse NeMo's scheduler from the original method when available.
        try:
            base = original_configure()
            if isinstance(base, dict) and "lr_scheduler" in base:
                base["optimizer"] = opt
                return base
            if isinstance(base, (list, tuple)) and len(base) == 2:
                _, schedulers = base
                return {"optimizer": opt, "lr_scheduler": schedulers[0]}
        except Exception:
            pass
        return opt

    model.configure_optimizers = configure_optimizers  # type: ignore[assignment]


# ---------- main ----------

def train() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NEMO.parent.mkdir(parents=True, exist_ok=True)
    train_manifest, val_manifest = _prepare_manifests()

    pl = _load_lightning()
    pl.seed_everything(SEED, workers=True)

    model = _load_model()
    model.setup_training_data(_data_cfg(train_manifest, shuffle=True, training=True))
    model.setup_validation_data(_data_cfg(val_manifest, shuffle=False, training=False))
    _configure_freeze(model)
    _enable_spec_augment(model)
    _set_optimizer(model)

    callbacks = []
    try:
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        callbacks.append(ModelCheckpoint(
            dirpath=str(RUN_DIR / "checkpoints"),
            monitor="val_wer",
            mode="min",
            save_top_k=2,
            filename="parakeet-{step:06d}-{val_wer:.4f}",
        ))
    except Exception as exc:
        print(f"warn: callbacks unavailable: {exc}")

    trainer_kwargs: dict[str, Any] = {
        "default_root_dir": str(RUN_DIR),
        "max_steps": MAX_STEPS,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "accumulate_grad_batches": GRAD_ACCUM,
        "log_every_n_steps": int(os.getenv("ASR_LOG_STEPS", "25")),
        "val_check_interval": int(os.getenv("ASR_EVAL_STEPS", "250")),
        "enable_checkpointing": True,
        "gradient_clip_val": float(os.getenv("ASR_GRAD_CLIP", "1.0")),
        "num_sanity_val_steps": 0,
        "callbacks": callbacks,
    }
    if torch.cuda.is_available() and _flag("ASR_MIXED_PRECISION", True):
        trainer_kwargs["precision"] = "bf16-mixed"  # bf16 if your GPU supports it; falls back to 16

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model)

    if OUTPUT_NEMO.exists() and _flag("ASR_BACKUP_EXISTING_NEMO", True):
        backup = OUTPUT_NEMO.with_suffix(".before_ft.nemo")
        if not backup.exists():
            OUTPUT_NEMO.replace(backup)
            print(f"backup -> {backup}")

    model.save_to(str(OUTPUT_NEMO))
    print(f"saved {OUTPUT_NEMO}")


if __name__ == "__main__":
    train()
