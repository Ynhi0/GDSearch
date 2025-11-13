"""
Fine-tune a Transformer (BERT) on IMDB with optimizer comparisons and gradient dynamics logging.

Outputs per-run CSVs compatible with the repository's result conventions.
This script guards optional dependencies (transformers, datasets) so import errors won't break CI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def _try_import_hf():
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
        from datasets import load_dataset
        return BertTokenizer, BertForSequenceClassification, load_dataset
    except Exception as e:
        raise RuntimeError(
            "HuggingFace 'transformers' and 'datasets' are required for this script. "
            "Install them via `pip install transformers datasets`."
        ) from e


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _flattened_grad_norm(model: torch.nn.Module) -> float:
    with torch.no_grad():
        grads = [p.grad.detach().view(-1) for n, p in model.named_parameters() if p.grad is not None]
        if not grads:
            return 0.0
        g = torch.cat(grads)
        return torch.linalg.norm(g, ord=2).item()


def _layer_grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            norms[name] = torch.linalg.norm(p.grad.view(-1), ord=2).item()
    return norms


def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    import torch.nn.functional as F
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += float(loss.item()) * input_ids.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def run_single_imdb(optimizer_name: str, seed: int, lr: float, epochs: int, batch_size: int, results_dir: Path):
    BertTokenizer, BertForSequenceClassification, load_dataset = _try_import_hf()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    # Data
    raw = load_dataset('imdb')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess(examples):
        enc = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)
        enc['labels'] = examples['label']
        return enc

    tokenized = raw.map(preprocess, batched=True)
    train_dataset = tokenized['train'].shuffle(seed=seed).select(range(2000))
    test_dataset = tokenized['test'].shuffle(seed=seed).select(range(1000))

    remove_cols = [c for c in train_dataset.column_names if c not in ('input_ids', 'attention_mask', 'labels')]
    train_dataset = train_dataset.remove_columns(remove_cols)
    test_dataset = test_dataset.remove_columns(remove_cols)

    def collate_fn(batch):
        import torch
        keys = batch[0].keys()
        out = {k: torch.tensor([b[k] for b in batch]) for k in keys}
        return out

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

    # Optimizer
    name = optimizer_name.upper()
    if name == 'ADAMW':
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
    elif name in ('SGD', 'SGD_MOMENTUM', 'SGD-MOMENTUM'):
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9 if 'MOMENTUM' in name else 0.0)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    history = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optim.zero_grad()
            loss.backward()
            grad_norm = _flattened_grad_norm(model)
            optim.step()
        # end of epoch: eval and layer grad norms (captured on last batch grads)
        test_loss, test_acc = evaluate(model, test_loader, device)
        layer_grads = _layer_grad_norms(model)
        row = {
            'epoch': epoch,
            'train_loss_last': float(loss.item()),
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'grad_norm': float(grad_norm),
        }
        # flatten a few representative layers for heterogeneity illustration
        for key in [k for k in layer_grads.keys() if 'encoder.layer.11' in k or 'classifier' in k][:8]:
            row[f'layer_grad[{key}]'] = layer_grads[key]
        history.append(row)

    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None

    df = pd.DataFrame(history)
    df['elapsed_seconds'] = elapsed
    df['peak_gpu_mb'] = peak_mb
    out_name = f"NN_BERT_IMDB_{optimizer_name}_lr{lr}_seed{seed}_application.csv"
    out_path = Path('results') / out_name
    Path('results').mkdir(exist_ok=True, parents=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='IMDB Transformer fine-tuning with optimizer comparison')
    parser.add_argument('--optimizers', type=str, default='AdamW,SGD_Momentum')
    parser.add_argument('--seeds', type=str, default='1,2,3')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr-adamw', type=float, default=5e-5)
    parser.add_argument('--lr-sgd', type=float, default=1e-3)
    args, _ = parser.parse_known_args()

    seeds = [int(s) for s in args.seeds.split(',') if s]
    opts = [o.strip() for o in args.optimizers.split(',') if o.strip()]
    for opt in opts:
        lr = args.lr_adamw if opt.upper().startswith('ADAMW') else args.lr_sgd
        for seed in seeds:
            try:
                run_single_imdb(opt, seed, lr, args.epochs, args.batch_size, Path('results'))
            except RuntimeError as e:
                print(str(e))
                return 1
            except Exception as e:
                print('Error:', e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())