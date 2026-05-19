"""Quick exact/public eval for a trained RAFT LoRA adapter.

This does not replace official evaluation. It helps catch broken adapters before
building a Docker image.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = os.getenv("RAFT_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER = Path(os.getenv("RAFT_ADAPTER_OUT", "/home/jupyter/nlp/src/lora_adapter"))
DATA = Path(os.getenv("RAFT_DATA_DIR", "/home/jupyter/nlp/raft_data")) / "eval.jsonl"
MAX_LEN = int(os.getenv("RAFT_MAX_MODEL_LEN", "2048"))


def norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()

    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    correct = 0
    for row in rows:
        inputs = tokenizer(row["prompt"], return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        correct += norm(row["answer"]) in norm(answer)
    print(f"contains-gold accuracy: {correct}/{len(rows)} = {correct / max(1, len(rows)):.3f}")


if __name__ == "__main__":
    main()
