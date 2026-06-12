"""Fine-tune Parakeet-TDT 0.6B on TIL-AI ASR training data.

Usage:
    python asr_finetune.py \
        --data_dir /home/jupyter/novice/asr \
        --output_dir /home/jupyter/asr_finetune \
        --epochs 5 \
        --batch_size 8

This script:
1. Reads asr.jsonl and creates NeMo-format manifests (train/val split)
2. Fine-tunes Parakeet-TDT with frozen encoder (decoder-only fine-tuning for speed)
3. Saves the best checkpoint for use in the ASR container
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import soundfile as sf


def create_manifests(data_dir: str, output_dir: str, val_ratio: float = 0.05):
    """Convert asr.jsonl to NeMo manifest format with train/val split."""
    jsonl_path = os.path.join(data_dir, "asr.jsonl")
    entries = []

    print(f"Reading {jsonl_path}...")
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            audio_path = os.path.join(data_dir, entry["audio"])

            if not os.path.exists(audio_path):
                print(f"  Skipping missing: {audio_path}")
                continue

            # Get duration from audio file
            try:
                info = sf.info(audio_path)
                duration = info.duration
            except Exception as e:
                print(f"  Error reading {audio_path}: {e}")
                continue

            entries.append({
                "audio_filepath": audio_path,
                "text": entry["transcript"].strip(),
                "duration": round(duration, 3),
            })

    print(f"Total valid entries: {len(entries)}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(entries)
    val_size = max(1, int(len(entries) * val_ratio))
    val_entries = entries[:val_size]
    train_entries = entries[val_size:]

    os.makedirs(output_dir, exist_ok=True)
    train_manifest = os.path.join(output_dir, "train_manifest.json")
    val_manifest = os.path.join(output_dir, "val_manifest.json")

    with open(train_manifest, "w") as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + "\n")

    with open(val_manifest, "w") as f:
        for entry in val_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Train: {len(train_entries)} samples -> {train_manifest}")
    print(f"Val:   {len(val_entries)} samples -> {val_manifest}")

    # Print some stats
    durations = [e["duration"] for e in entries]
    print(f"Duration stats: min={min(durations):.1f}s, max={max(durations):.1f}s, "
          f"mean={sum(durations)/len(durations):.1f}s, total={sum(durations)/3600:.1f}h")

    return train_manifest, val_manifest


def finetune(
    train_manifest: str,
    val_manifest: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    freeze_encoder: bool = True,
):
    """Fine-tune Parakeet-TDT on the training data."""
    import pytorch_lightning as pl
    import torch
    import nemo.collections.asr as nemo_asr
    from nemo.utils import exp_manager

    print("\n=== Loading Parakeet-TDT 0.6B ===")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

    # Freeze encoder for faster training (only fine-tune decoder + joint)
    if freeze_encoder:
        print("Freezing encoder (decoder-only fine-tuning for speed)")
        if hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = False
        # Also freeze preprocessor
        if hasattr(model, 'preprocessor'):
            for param in model.preprocessor.parameters():
                param.requires_grad = False

    # Update training data config
    model.cfg.train_ds.manifest_filepath = train_manifest
    model.cfg.train_ds.batch_size = batch_size
    model.cfg.train_ds.num_workers = 4
    model.cfg.train_ds.pin_memory = True
    model.cfg.train_ds.shuffle = True

    # Filter out very long audio (>35s) to avoid OOM on T4
    if hasattr(model.cfg.train_ds, 'max_duration'):
        model.cfg.train_ds.max_duration = 35.0
    if hasattr(model.cfg.train_ds, 'min_duration'):
        model.cfg.train_ds.min_duration = 0.5

    # Update validation data config
    model.cfg.validation_ds.manifest_filepath = val_manifest
    model.cfg.validation_ds.batch_size = batch_size
    model.cfg.validation_ds.num_workers = 4

    # Setup data loaders
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    # Configure optimizer
    model.cfg.optim.name = "adamw"
    model.cfg.optim.lr = lr
    model.cfg.optim.weight_decay = 0.01

    # Scheduler: warmup then cosine decay
    model.cfg.optim.sched = {
        "name": "CosineAnnealing",
        "warmup_steps": 100,
        "min_lr": 1e-6,
    }

    # Setup trainer
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=epochs,
        accumulate_grad_batches=4,  # Effective batch = batch_size * 4
        precision="16-mixed",  # fp16 for T4
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        default_root_dir=output_dir,
        gradient_clip_val=1.0,
    )

    # Setup experiment manager for checkpointing
    exp_config = {
        "exp_dir": output_dir,
        "name": "parakeet_finetune",
        "checkpoint_callback_params": {
            "monitor": "val_wer",
            "mode": "min",
            "save_top_k": 3,
            "save_last": True,
        },
        "create_wandb_logger": False,
        "create_tensorboard_logger": True,
    }
    exp_manager.exp_manager(trainer, exp_config)

    print(f"\n=== Starting fine-tuning ===")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} (effective: {batch_size * 4})")
    print(f"  Learning rate: {lr}")
    print(f"  Encoder frozen: {freeze_encoder}")
    print(f"  Output: {output_dir}")

    # Train
    trainer.fit(model)

    # Save the final model as .nemo
    best_model_path = os.path.join(output_dir, "parakeet_finetuned.nemo")
    model.save_to(best_model_path)
    print(f"\n=== Fine-tuning complete! ===")
    print(f"Best model saved to: {best_model_path}")
    print(f"\nTo use in BrainHack_V2:")
    print(f"  cp {best_model_path} ~/BrainHack_V2/asr/model/parakeet_finetuned.nemo")
    print(f"  # Then update asr_manager.py to load from /app/model/parakeet_finetuned.nemo")

    return best_model_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Parakeet-TDT on TIL ASR data")
    parser.add_argument("--data_dir", default="/home/jupyter/novice/asr",
                        help="Directory containing asr.jsonl and WAV files")
    parser.add_argument("--output_dir", default="/home/jupyter/asr_finetune",
                        help="Output directory for manifests and checkpoints")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (effective = batch_size * 4 due to grad accumulation)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--freeze_encoder", action="store_true", default=True,
                        help="Freeze encoder (faster, decoder-only fine-tuning)")
    parser.add_argument("--full_finetune", action="store_true", default=False,
                        help="Fine-tune entire model (slower but potentially better)")
    parser.add_argument("--manifest_only", action="store_true", default=False,
                        help="Only create manifests, don't train")
    args = parser.parse_args()

    # Step 1: Create manifests
    train_manifest, val_manifest = create_manifests(args.data_dir, args.output_dir)

    if args.manifest_only:
        print("Manifest-only mode, skipping training.")
        return

    # Step 2: Fine-tune
    freeze = not args.full_finetune
    finetune(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_encoder=freeze,
    )


if __name__ == "__main__":
    main()
