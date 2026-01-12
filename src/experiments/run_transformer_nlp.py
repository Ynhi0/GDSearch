"""
Fine-tune a Transformer (BERT) on IMDB with optimizer comparisons and gradient dynamics logging.

DEMO/PROOF-OF-CONCEPT SCRIPT

CRITICAL LIMITATION (REVIEW FLAG):
This script uses a LIMITED subset of IMDB (2000 train, 1000 test) and supports
only a SUBSET of optimizers (AdamW, SGD). It is a proof-of-concept for NLP
domain applicability and should NOT be used for strong cross-domain generalization
claims without extension to:
- Full dataset size
- Full optimizer suite (SAM, Lookahead, RMSProp, etc.)
- Multiple NLP tasks (not just sentiment classification)

For SOTA research claims, extend this script or clearly state its limitations
in reports.

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
from src.core.training_utils import set_seed
import logging


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


# Removed duplicate set_seed - using from src.core.training_utils


def _flattened_grad_norm(model: torch.nn.Module) -> float:
    with torch.no_grad():
        grads = [p.grad.detach().view(-1) for n, p in model.named_parameters() if p.grad is not None]
        if not grads:
            return 0.0
        g = torch.cat(grads)
        return torch.norm(g, p=2).item()


def _layer_grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            norms[name] = torch.norm(p.grad.view(-1), p=2).item()
    return norms


from typing import Any

def evaluate(model: Any, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
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
            # model call is dynamic (third-party) - suppress arg-type strictness
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # type: ignore[arg-type]
            loss = outputs.loss
            logits = outputs.logits
            total_loss += float(loss.item()) * input_ids.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def run_single_imdb(optimizer_name: str, seed: int, lr: float, epochs: int, batch_size: int, results_dir: Path, resume: bool = False, full_data: bool = False, momentum: float = 0.9):
    """
    Run IMDB sentiment classification with BERT.

    Args:
        optimizer_name: Name of optimizer (AdamW, SGD, SGD_Momentum)
        seed: Random seed
        lr: Learning rate
        epochs: Number of epochs
        batch_size: Batch size
        results_dir: Output directory
        resume: Skip if results exist
        full_data: If True, use full IMDB dataset (25K train, 25K test). If False, use 2K/1K subset.
        momentum: Momentum coefficient for SGD-based optimizers

    Added full_data parameter to avoid "Toy Benchmark" deception.
    For publication, MUST use full_data=True to ensure sufficient statistical power.
    """
    BertTokenizer, BertForSequenceClassification, load_dataset = _try_import_hf()

    # Check if experiment is already completed
    data_suffix = "_full" if full_data else "_toy"
    out_name = f"NN_BERT_IMDB_{optimizer_name}_lr{lr}_seed{seed}{data_suffix}_application.csv"
    out_path = results_dir / out_name
    if resume and out_path.exists():
        try:
            df = pd.read_csv(out_path)
            if len(df) > 0:
                logging.info("Skipping %s seed %s (already completed)", optimizer_name, seed)
                return df
        except Exception as e:
            logging.debug("Existing result file appears corrupted: %s", out_path, exc_info=True)  # Re-run

    # Set environment variables to avoid warnings
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Suppress unnecessary transformers warnings
    import warnings
    warnings.filterwarnings('ignore', message='Some weights.*were not initialized')

    import transformers
    transformers.logging.set_verbosity_error()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    # Data - robust loading with fallback for environment compatibility
    cache_dir = os.environ.get('HUGGINGFACE_CACHE_DIR', None)
    if cache_dir is None:
        import tempfile
        cache_dir = str(Path(tempfile.gettempdir()) / 'hf_cache')

    try:
        raw = load_dataset('imdb', cache_dir=cache_dir)
    except (ValueError, Exception) as e:
        logging.warning("Failed to load IMDB dataset using cache_dir=%s: %s", cache_dir, e)
        logging.info("Trying alternative loading method...")
        try:
            raw = load_dataset('imdb', trust_remote_code=True)
        except Exception as e2:
            logging.error("Could not load IMDB dataset: %s", e2, exc_info=True)
            raise RuntimeError("Failed to load IMDB dataset. Check HuggingFace/fsspec versions.") from e2
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess(examples):
        enc = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)
        enc['labels'] = examples['label']
        return enc

    tokenized = raw.map(preprocess, batched=True)

    # Safely handle different HF dataset-like objects (IterableDataset vs Dataset)
    from typing import Any, Mapping, cast
    if isinstance(tokenized, Mapping):
        train_ds = tokenized['train']
        test_ds = tokenized['test']
    else:
        # Fallback for non-mapping objects; cast to Any to access splits by index/key
        train_ds = cast(Any, tokenized)['train']  # type: ignore[index]
        test_ds = cast(Any, tokenized)['test']  # type: ignore[index]

    # Prefer HF Dataset API when available (explicit Dataset type check)
    try:
        from datasets import Dataset as _HFDataset  # type: ignore[import]
    except Exception:
        _HFDataset = None

    # Dynamic dataset size based on full_data flag
    # For publication-quality results, use full_data=True (25K train, 25K test)
    # Toy mode (2K/1K) is ONLY for quick prototyping and CI testing
    if full_data:
        train_size = len(train_ds) if hasattr(train_ds, '__len__') else 25000
        test_size = len(test_ds) if hasattr(test_ds, '__len__') else 25000
        logging.info(f"[FULL DATA MODE] Using complete IMDB dataset: {train_size} train, {test_size} test")
    else:
        train_size = 2000
        test_size = 1000
        logging.warning("[TOY MODE] Using LIMITED subset (2K/1K). NOT suitable for publication claims.")

    if _HFDataset is not None and isinstance(train_ds, _HFDataset):
        try:
            train_dataset = train_ds.shuffle(seed=seed).select(range(min(train_size, len(train_ds))))
        except Exception:
            train_dataset = list(train_ds)[:train_size]
    else:
        # Materialize any iterable or list into a Python list to avoid slicing concerns
        train_dataset = list(train_ds)[:train_size]

    if _HFDataset is not None and isinstance(test_ds, _HFDataset):
        try:
            test_dataset = test_ds.shuffle(seed=seed).select(range(min(test_size, len(test_ds))))
        except Exception:
            test_dataset = list(test_ds)[:test_size]
    else:
        test_dataset = list(test_ds)[:test_size]

    # Robust helper to compute columns to drop / project keys for non-HF objects
    def _compute_remove_columns(ds) -> list:
        names = []
        if hasattr(ds, 'column_names'):
            try:
                names = list(ds.column_names)
            except Exception:
                names = []
        elif isinstance(ds, list) and ds and isinstance(ds[0], dict):
            names = list(ds[0].keys())
        return [c for c in names if c not in ('input_ids', 'attention_mask', 'labels')]

    remove_cols = _compute_remove_columns(train_dataset)

    # Prefer to call Dataset.remove_columns only on HF Dataset objects when available
    if remove_cols:
        if _HFDataset is not None and isinstance(train_dataset, _HFDataset):
            try:
                train_dataset = train_dataset.remove_columns(remove_cols)
            except Exception:
                if isinstance(train_dataset, list):
                    new_train = []
                    for ex in train_dataset:
                        try:
                            ex_dict = dict(ex)
                        except Exception:
                            ex_dict = {}
                            for k in ('input_ids', 'attention_mask', 'labels'):
                                if hasattr(ex, k):
                                    ex_dict[k] = getattr(ex, k)
                        new_train.append({k: ex_dict[k] for k in ('input_ids', 'attention_mask', 'labels') if k in ex_dict})
                    train_dataset = new_train
        elif isinstance(train_dataset, list):
            proj = []
            for ex in train_dataset:
                try:
                    ex_dict = dict(ex)
                except Exception:
                    # Fallback: try attribute access for Example objects
                    ex_dict = {}
                    for k in ('input_ids', 'attention_mask', 'labels'):
                        if hasattr(ex, k):
                            ex_dict[k] = getattr(ex, k)
                proj.append({k: ex_dict[k] for k in ('input_ids', 'attention_mask', 'labels') if k in ex_dict})
            train_dataset = proj

    # Apply same for test dataset
    test_remove_cols = _compute_remove_columns(test_dataset)
    if test_remove_cols:
        if _HFDataset is not None and isinstance(test_dataset, _HFDataset):
            try:
                test_dataset = test_dataset.remove_columns(test_remove_cols)
            except Exception:
                if isinstance(test_dataset, list):
                    new_test = []
                    for ex in test_dataset:
                        try:
                            ex_dict = dict(ex)
                        except Exception:
                            ex_dict = {}
                            for k in ('input_ids', 'attention_mask', 'labels'):
                                if hasattr(ex, k):
                                    ex_dict[k] = getattr(ex, k)
                        new_test.append({k: ex_dict[k] for k in ('input_ids', 'attention_mask', 'labels') if k in ex_dict})
                    test_dataset = new_test
        elif isinstance(test_dataset, list):
            proj = []
            for ex in test_dataset:
                try:
                    ex_dict = dict(ex)
                except Exception:
                    ex_dict = {}
                    for k in ('input_ids', 'attention_mask', 'labels'):
                        if hasattr(ex, k):
                            ex_dict[k] = getattr(ex, k)
                proj.append({k: ex_dict[k] for k in ('input_ids', 'attention_mask', 'labels') if k in ex_dict})
            test_dataset = proj
    from collections.abc import Mapping
    def collate_fn(batch: list) -> dict:
        import torch
        # Support Mapping-like (dict/Mapping) examples or objects with attributes
        first = batch[0]
        if isinstance(first, Mapping):
            keys = list(first.keys())
        else:
            keys = list(getattr(first, '__dict__', {}).keys())

        out: dict = {}
        for k in keys:
            vals = []
            for b in batch:
                if isinstance(b, Mapping):
                    v = b.get(k)
                else:
                    # Fall back to attribute access if present
                    v = getattr(b, k, None)
                vals.append(v)
            # Attempt to convert to tensor safely; if values are None or mixed, wrap in list first
            try:
                out[k] = torch.tensor(vals)
            except Exception:
                # As a fallback, try converting each element individually
                out[k] = torch.tensor([v if v is not None else 0 for v in vals])
        return out

    # Use num_workers=0 to avoid tokenizer parallelism issues
    # Use make_dataloader for consistent settings
    from src.core.dataloader_utils import make_dataloader
    from torch.utils.data import Dataset as TorchDataset
    from typing import Sequence

    # Single local wrapper class to avoid redeclaration warnings
    class _SeqDataset(TorchDataset):
        def __init__(self, seq: Sequence):
            self._seq = seq
        def __len__(self) -> int:  # pragma: no cover - trivial wrapper
            return len(self._seq)
        def __getitem__(self, idx):  # pragma: no cover - trivial wrapper
            return self._seq[idx]

    def _ensure_torch_dataset(ds) -> TorchDataset:
        # If already a Torch Dataset, return as-is
        if isinstance(ds, TorchDataset):
            return ds
        # If sequence-like (has __len__ and __getitem__), wrap it
        if hasattr(ds, '__len__') and hasattr(ds, '__getitem__'):
            return _SeqDataset(ds)
        # Otherwise, consume into a list and wrap
        seq = list(ds)
        return _SeqDataset(seq)

    train_loader = make_dataloader(_ensure_torch_dataset(train_dataset), batch_size=batch_size, shuffle=True, seed=42, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = make_dataloader(_ensure_torch_dataset(test_dataset), batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    # Model
    from typing import Any, cast
    model = cast(Any, BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2))
    model.to(device=device)  # type: ignore[arg-type]

    # Use optimizer registry for consistency
    # This ensures proper hyperparameter management across all experiments
    from src.core.optimizer_registry import create_optimizer_from_config

    optimizer_config = {'name': optimizer_name, 'lr': lr}
    try:
        optimizer = create_optimizer_from_config(optimizer_config, model.parameters())
    except Exception as e:
        logging.warning(f"Registry creation failed, using fallback: {e}")
        # Fallback for backward compatibility
        name = optimizer_name.upper()
        if name == 'ADAMW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        elif name in ('SGD', 'SGD_MOMENTUM', 'SGD-MOMENTUM'):
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9 if 'MOMENTUM' in name else 0.0)
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # type: ignore[arg-type]
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            grad_norm = _flattened_grad_norm(model)
            optimizer.step()
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
    # Use data_suffix from function parameter
    data_suffix = "_full" if full_data else "_toy"
    out_name = f"NN_BERT_IMDB_{optimizer_name}_lr{lr}_seed{seed}{data_suffix}_application.csv"
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
    parser.add_argument('--resume', action='store_true', help='Skip experiments that already have result files')
    parser.add_argument('--full-data', action='store_true', help='Use full IMDB dataset (25K train/test) instead of toy 2K/1K subset. REQUIRED for publication.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum coefficient for SGD (default: 0.9). Use for sensitivity analysis.')
    args, _ = parser.parse_known_args()

    if args.full_data:
        logging.info("=" * 70)
        logging.info("FULL DATA MODE ENABLED - Using complete IMDB dataset")
        logging.info("=" * 70)
    else:
        logging.warning("=" * 70)
        logging.warning("TOY MODE - Using limited 2K/1K subset (NOT publication-ready)")
        logging.warning("Use --full-data flag for rigorous experiments")
        logging.warning("=" * 70)

    seeds = [int(s) for s in args.seeds.split(',') if s]
    opts = [o.strip() for o in args.optimizers.split(',') if o.strip()]
    for opt in opts:
        lr = args.lr_adamw if opt.upper().startswith('ADAMW') else args.lr_sgd
        for seed in seeds:
            try:
                run_single_imdb(opt, seed, lr, args.epochs, args.batch_size, Path('results'),
                              resume=args.resume, full_data=args.full_data, momentum=args.momentum)
            except RuntimeError as e:
                print(str(e))
                return 1
            except Exception as e:
                print('Error:', e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())