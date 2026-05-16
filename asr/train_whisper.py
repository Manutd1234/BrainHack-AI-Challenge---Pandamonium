"""Fine-tune Whisper on the novice ASR WAV/transcript set.

Run in Jupyter:

    cd ~/asr
    pip install -r requirements-train.txt
    python train_whisper.py

The trained model is saved to ./model/whisper-small-til26 and is copied into
the Docker image by the ASR Dockerfile.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import librosa
import soundfile as sf
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


DATA_DIR = Path(os.getenv("ASR_DATA_DIR", "/home/jupyter/novice/asr"))
MODEL_ID = os.getenv("WHISPER_BASE_MODEL_ID", "openai/whisper-small")
OUTPUT_DIR = Path(
    os.getenv("WHISPER_OUTPUT_DIR", "/home/jupyter/asr/model/whisper-small-til26")
)
MAX_STEPS = int(os.getenv("ASR_TRAIN_STEPS", "1500"))
EVAL_SIZE = float(os.getenv("ASR_EVAL_SIZE", "0.05"))
SEED = int(os.getenv("ASR_SEED", "26"))


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _audio_field(item: dict[str, Any]) -> str:
    for key in ("audio", "file", "path", "wav"):
        if key in item and item[key] is not None:
            return str(item[key])
    raise KeyError(f"No audio path field in item: {item.keys()}")


def _transcript_field(item: dict[str, Any]) -> str:
    for key in ("transcript", "text", "sentence"):
        if key in item and item[key] is not None:
            return str(item[key])
    raise KeyError(f"No transcript field in item: {item.keys()}")


def load_dataset() -> Dataset:
    rows = []
    with (DATA_DIR / "asr.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            transcript = _transcript_field(item).strip()
            if not transcript:
                continue
            rows.append(
                {
                    "audio": str(DATA_DIR / _audio_field(item)),
                    "sentence": transcript,
                    "language": str(item.get("language", "")),
                }
            )

    return Dataset.from_list(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset().train_test_split(test_size=EVAL_SIZE, seed=SEED)

    processor = WhisperProcessor.from_pretrained(MODEL_ID, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    def prepare(batch: dict[str, Any]) -> dict[str, Any]:
        audio_array, sample_rate = sf.read(batch["audio"], dtype="float32", always_2d=False)
        if getattr(audio_array, "ndim", 1) == 2:
            audio_array = audio_array.mean(axis=1)
        if sample_rate != 16_000:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sample_rate,
                target_sr=16_000,
            )
        batch["input_features"] = processor.feature_extractor(
            audio_array,
            sampling_rate=16_000,
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    dataset = dataset.map(
        prepare,
        remove_columns=dataset["train"].column_names,
        num_proc=int(os.getenv("ASR_PREPROCESS_WORKERS", "1")),
    )

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=int(os.getenv("ASR_TRAIN_BATCH", "8")),
        per_device_eval_batch_size=int(os.getenv("ASR_EVAL_BATCH", "8")),
        gradient_accumulation_steps=int(os.getenv("ASR_GRAD_ACCUM", "2")),
        learning_rate=float(os.getenv("ASR_LR", "1e-5")),
        warmup_steps=int(os.getenv("ASR_WARMUP_STEPS", "100")),
        max_steps=MAX_STEPS,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=int(os.getenv("ASR_EVAL_STEPS", "250")),
        save_steps=int(os.getenv("ASR_SAVE_STEPS", "250")),
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=False,
        report_to=[],
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved fine-tuned Whisper model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
