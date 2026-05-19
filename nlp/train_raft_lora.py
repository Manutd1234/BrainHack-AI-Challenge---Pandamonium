"""Train a compact Qwen LoRA adapter on RAFT examples.

This is optional. Keep the production RAG baseline safe, then copy the trained
adapter into nlp/src/lora_adapter and enable NLP_USE_LLM=1 for an experiment.

Run on Jupyter after build_raft_data.py:

    cd ~/nlp
    pip install -r requirements-train.txt
    RAFT_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct python train_raft_lora.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


MODEL_ID = os.getenv("RAFT_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
DATA_DIR = Path(os.getenv("RAFT_DATA_DIR", "/home/jupyter/nlp/raft_data"))
OUT_DIR = Path(os.getenv("RAFT_ADAPTER_OUT", "/home/jupyter/nlp/src/lora_adapter"))
MAX_LEN = int(os.getenv("RAFT_MAX_MODEL_LEN", "2048"))
STEPS = int(os.getenv("RAFT_TRAIN_STEPS", "900"))
BATCH = int(os.getenv("RAFT_BATCH", "1"))
GRAD_ACCUM = int(os.getenv("RAFT_GRAD_ACCUM", "8"))
LR = float(os.getenv("RAFT_LR", "2e-4"))


def main() -> None:
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "eval": str(DATA_DIR / "eval.jsonl"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=int(os.getenv("RAFT_LORA_R", "16")),
        lora_alpha=int(os.getenv("RAFT_LORA_ALPHA", "32")),
        lora_dropout=float(os.getenv("RAFT_LORA_DROPOUT", "0.05")),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=os.getenv(
            "RAFT_TARGET_MODULES",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        ).split(","),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
        return encoded

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=dataset["train"].column_names)
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        max_steps=STEPS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"saved LoRA adapter to {OUT_DIR}")


if __name__ == "__main__":
    main()
