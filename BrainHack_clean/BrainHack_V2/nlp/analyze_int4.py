"""Small helper to estimate whether a Qwen model can fit as INT4/QLoRA."""

from __future__ import annotations

import os

from transformers import AutoConfig


MODEL_ID = os.getenv("RAFT_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    params = getattr(config, "num_parameters", None)
    if callable(params):
        total = params()
    else:
        hidden = getattr(config, "hidden_size", 0)
        layers = getattr(config, "num_hidden_layers", 0)
        vocab = getattr(config, "vocab_size", 0)
        total = 12 * layers * hidden * hidden + vocab * hidden
    int4_gb = total * 0.5 / (1024**3)
    bf16_gb = total * 2.0 / (1024**3)
    print(f"model={MODEL_ID}")
    print(f"rough params={total/1e9:.2f}B")
    print(f"rough INT4 weights={int4_gb:.2f} GiB")
    print(f"rough BF16 weights={bf16_gb:.2f} GiB")


if __name__ == "__main__":
    main()
