#!/usr/bin/env python3
"""
Kaggle-ready IMDB publication experiments (DistilBERT).
- Multi-seed runs across AdamW and SGD_Momentum
- Per-epoch metrics, telemetry (elapsed_seconds, peak_gpu_mb)
- Saves per-run CSVs and a paired statistical comparison CSV (Holm–Bonferroni)

This script is standalone (no repository imports) for easy Kaggle usage.
Requires GPU + Internet (datasets, model weights).
"""
import os
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import torch
from torch.utils.data import DataLoader


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


def _try_import_hf():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from datasets import load_dataset
        return AutoTokenizer, AutoModelForSequenceClassification, load_dataset
    except Exception as e:
        raise RuntimeError(
            "This script requires `transformers` and `datasets`. Install via `pip install transformers datasets accelerate`."
        ) from e


def collate_fn_builder(tokenizer):
    def collate_fn(examples):
        import torch
        input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
        attention_mask = [torch.tensor(ex.get("attention_mask", [])) for ex in examples]
        labels = [torch.tensor(ex["label"]) for ex in examples]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        if attention_mask and len(attention_mask[0]) > 0:
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        else:
            attention_mask = None
        labels = torch.stack(labels)
        batch = {"input_ids": input_ids, "labels": labels}
        if attention_mask is not None:
            batch["attention_mask"] = attention_mask
        return batch
    return collate_fn


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * input_ids.size(0)
        total += input_ids.size(0)
    return total_loss / max(1, total)


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += float(loss.item()) * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def _ckpt_path(ckpt_dir: Path, opt_name: str, seed: int, lr: float, model_name: str) -> Path:
    safe_model = model_name.replace('/', '_')
    return ckpt_dir / f"IMDB_{safe_model}_{opt_name}_lr{lr}_seed{seed}.pt"


def run_single(opt_name: str, seed: int, lr: float, epochs: int, batch_size: int, model_name: str, results_dir: Path, train_size: int, test_size: int, resume: bool, ckpt_dir: Path):
    AutoTokenizer, AutoModel, load_dataset = _try_import_hf()
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw = load_dataset('imdb')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=False, max_length=256)

    tokenized = raw.map(preprocess, batched=True)
    train_ds = tokenized['train'].shuffle(seed=seed).select(range(min(train_size, len(tokenized['train']))))
    test_ds = tokenized['test'].shuffle(seed=seed).select(range(min(test_size, len(tokenized['test']))))

    # keep only needed columns
    keep = ['input_ids', 'attention_mask', 'label']
    rm_train = [c for c in train_ds.column_names if c not in keep]
    rm_test = [c for c in test_ds.column_names if c not in keep]
    train_ds = train_ds.remove_columns(rm_train)
    test_ds = test_ds.remove_columns(rm_test)

    collate_fn = collate_fn_builder(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = AutoModel.from_pretrained(model_name, num_labels=2).to(device)

    name = opt_name.upper()
    if name.startswith('ADAMW'):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif name in ('SGD', 'SGD_MOMENTUM', 'SGD-MOMENTUM'):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9 if 'MOMENTUM' in name else 0.0)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    history = []
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = _ckpt_path(ckpt_dir, opt_name, seed, lr, model_name)
    start_epoch = 1
    # Resume logic
    if resume and ckpt_file.exists():
        try:
            state = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(state['model'], strict=False)
            if state.get('opt_name', '').upper().startswith('ADAMW') and isinstance(optimizer, torch.optim.AdamW):
                optimizer.load_state_dict(state['optimizer'])
            if state.get('opt_name', '').upper().startswith('SGD') and isinstance(optimizer, torch.optim.SGD):
                optimizer.load_state_dict(state['optimizer'])
            start_epoch = int(state.get('epoch', 0)) + 1
            history = state.get('history', [])
            print(f"Resuming from epoch {start_epoch} using checkpoint: {ckpt_file}")
        except Exception as e:
            print('Resume failed:', e)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'test_loss': te_loss, 'test_acc': te_acc})
        print(f"seed={seed} {opt_name} [{epoch}/{epochs}] train_loss={tr_loss:.4f} test_acc={te_acc:.3f}")
        # Save checkpoint each epoch (last-writer-wins)
        try:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'history': history,
                'opt_name': opt_name,
                'seed': seed,
                'lr': lr,
                'model_name': model_name,
            }, ckpt_file)
        except Exception as e:
            print('Warning: failed to save checkpoint:', e)

    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
    df = pd.DataFrame(history)
    df['elapsed_seconds'] = elapsed
    df['peak_gpu_mb'] = peak_mb
    out = results_dir / f"NN_DistilBERT_IMDB_{opt_name}_lr{lr}_seed{seed}_publication.csv"
    df.to_csv(out, index=False)
    return out


def compute_statistics(results_dir: str):
    import glob, re
    patterns = {
        'AdamW': f"{results_dir}/NN_DistilBERT_IMDB_AdamW_*_publication.csv",
        'SGD_Momentum': f"{results_dir}/NN_DistilBERT_IMDB_SGD_Momentum_*_publication.csv",
    }
    data = {}
    for opt, pat in patterns.items():
        vals = {}
        for f in glob.glob(pat):
            m = re.search(r"seed(\d+)", f)
            if not m:
                continue
            seed = int(m.group(1))
            df = pd.read_csv(f)
            vals[seed] = float(df['test_acc'].iloc[-1]) if 'test_acc' in df.columns else float('nan')
        data[opt] = vals
    rows = []
    A, B = 'AdamW', 'SGD_Momentum'
    common = sorted(set(data.get(A, {}).keys()) & set(data.get(B, {}).keys()))
    if len(common) >= 3:
        a = np.array([data[A][s] for s in common])
        b = np.array([data[B][s] for s in common])
        _, pA = stats.shapiro(a)
        _, pB = stats.shapiro(b)
        if pA > 0.05 and pB > 0.05:
            test = 'Paired t-test'
            stat, p = stats.ttest_rel(a, b)
            eff_name = "Cohen's d"
            eff = (a - b).mean() / (a - b).std(ddof=1)
        else:
            test = 'Wilcoxon'
            W, p = stats.wilcoxon(a, b)
            n = len(a)
            eff_name = 'Rank-biserial r'
            eff = 1 - (2 * W) / (n * (n + 1))
        rows.append({
            'Optimizer A': A, 'Optimizer B': B, 'n': len(common),
            'Mean A': float(a.mean()), 'Mean B': float(b.mean()), 'Test': test,
            'p-value': float(p), 'Effect': f"{eff_name}={eff:.3f}",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        out = Path(results_dir) / 'imdb_statistical_comparisons_publication.csv'
        df.to_csv(out, index=False)
        print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description='IMDB (DistilBERT) Kaggle Publication Suite')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--lr-adamw', type=float, default=5e-5)
    parser.add_argument('--lr-sgd', type=float, default=1e-3)
    parser.add_argument('--train-size', type=int, default=5000)
    parser.add_argument('--test-size', type=int, default=2000)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint if available')
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints')
    args, _ = parser.parse_known_args()

    if args.quick:
        seeds = [1, 2, 3]
        epochs = 2
        train_size = 1000
        test_size = 1000
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        epochs = args.epochs
        train_size = args.train_size
        test_size = args.test_size

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)

    configs = [
        ('AdamW', args.lr_adamw),
        ('SGD_Momentum', args.lr_sgd),
    ]

    total = len(configs) * len(seeds)
    print(f"Total runs: {total}")

    for opt, lr in configs:
        for seed in seeds:
            try:
                run_single(opt, seed, lr, epochs, args.batch_size, args.model_name, results_dir, train_size, test_size, args.resume, ckpt_dir)
            except Exception as e:
                print('Error:', e)
    compute_statistics(str(results_dir))


if __name__ == '__main__':
    main()
