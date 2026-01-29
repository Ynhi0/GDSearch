"""Small QLoRA training skeleton (poC)

This script provides a minimal, documented CLI to run QLoRA-style finetuning using
transformers + peft + bitsandbytes. For the PoC we implement a dry-run mode that
validates inputs without running long jobs. Fill in Colab runtime configs before
actual runs (GPU type, memory, and accelerate config).
"""

from dataclasses import dataclass
import argparse
import json
import sys
from pathlib import Path


@dataclass
class Config:
    data_path: str
    output_dir: str
    base_model: str = "meta-llama/Llama-2-7b-chat-hf"
    epochs: int = 1
    batch_size: int = 8
    lr: float = 2e-4


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning skeleton for BD-NSCA PoC")
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    return p.parse_args()


def load_data(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")
    # Minimal loader: read JSONL and return a list
    data = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            data.append(json.loads(line))
    return data


def main_cli():
    args = parse_args()
    cfg = Config(data_path=args.data_path, output_dir=args.output_dir, base_model=args.base_model)

    if args.dry_run:
        print("Dry run: validating inputs...")
        # Validate data
        data = load_data(cfg.data_path)
        print(f"Loaded {len(data)} examples. Base model: {cfg.base_model}")
        print("Dry run OK. Exiting.")
        return 0

    # Real training pipeline (placeholder):
    # 1. Tokenize / preprocess
    # 2. Initialize bitsandbytes 8-bit quantized model
    # 3. Create PEFT LoRA config and Trainer
    # 4. Train with accelerate / f16 mixed precision
    # 5. Save adapter and merged weights

    # NOTE: replace the following pseudo-code with concrete training steps in Colab
    print("Starting placeholder training loop (replace with real training steps in Colab)")

    # Pseudo steps (not executed):
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # from peft import LoraConfig, get_peft_model
    # tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    # model = AutoModelForCausalLM.from_pretrained(cfg.base_model, load_in_8bit=True, device_map='auto')
    # lora_config = LoraConfig(...)
    # model = get_peft_model(model, lora_config)
    # trainer = Trainer(...)
    # trainer.train()

    print("Training complete (placeholder). Save adapters to", cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())