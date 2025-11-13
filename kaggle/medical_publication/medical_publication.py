#!/usr/bin/env python3
"""
Kaggle-ready 2D medical segmentation publication experiments.
- Loads 2D images and binary masks from a dataset under /kaggle/input/<name> (images/, masks/)
- Trains a small 2D U-Net
- Multi-seed comparison of Adam vs SGD+Momentum
- Logs per-epoch Dice and per-run telemetry; writes per-run CSV and a stats CSV

This script is standalone (no repository imports) for easy Kaggle usage.
"""
import os
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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


class SegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, size=256):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.size = size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB').resize((self.size, self.size))
        mask = Image.open(self.mask_paths[idx]).convert('L').resize((self.size, self.size))
        img = np.asarray(img, dtype=np.float32) / 255.0
        mask = np.asarray(mask, dtype=np.float32)
        # binarize
        mask = (mask > 127.5).astype(np.float32)
        # to CHW tensors
        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask)[None, ...]
        return img, mask


def dice_coeff(pred, target, eps=1e-6):
    # pred, target in [0,1], shape [B,1,H,W]
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_channels, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*4, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.u3c = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.u2c = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.u1c = DoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.p1(c1))
        c3 = self.d3(self.p2(c2))
        bn = self.bottleneck(self.p3(c3))
        u3 = self.u3(bn)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.u3c(u3)
        u2 = self.u2(u3)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.u2c(u2)
        u1 = self.u1(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.u1c(u1)
        logits = self.outc(u1)
        return logits


def split_dataset(root: str, val_frac=0.1):
    img_dir = Path(root) / 'images'
    mask_dir = Path(root) / 'masks'
    img_paths = sorted([p for p in img_dir.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    mask_paths = sorted([p for p in mask_dir.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    n = min(len(img_paths), len(mask_paths))
    if n == 0:
        return [], [], [], []
    img_paths = img_paths[:n]
    mask_paths = mask_paths[:n]
    idx = int(n * (1 - val_frac))
    return img_paths[:idx], mask_paths[:idx], img_paths[idx:], mask_paths[idx:]


def synthetic_dataset(n=20, size=128):
    # simple blobs as images and masks
    imgs, masks = [], []
    for i in range(n):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        # draw a random circle
        rr, cc = np.ogrid[:size, :size]
        cx, cy = np.random.randint(size//4, 3*size//4, size=2)
        r = np.random.randint(size//8, size//5)
        circle = (rr - cx) ** 2 + (cc - cy) ** 2 <= r ** 2
        img[..., 1][circle] = 255
        mask[circle] = 255
        imgs.append(Image.fromarray(img))
        masks.append(Image.fromarray(mask))
    # save to temp folder
    tmp = Path('synthetic_medseg')
    (tmp / 'images').mkdir(parents=True, exist_ok=True)
    (tmp / 'masks').mkdir(parents=True, exist_ok=True)
    for i, (im, mk) in enumerate(zip(imgs, masks)):
        im.save(tmp / 'images' / f'{i:04d}.png')
        mk.save(tmp / 'masks' / f'{i:04d}.png')
    return str(tmp)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = bce(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
    return total_loss / max(1, len(loader.dataset))


def evaluate(model, loader, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()
            dices.append(dice_coeff(pred, y))
    return float(np.mean(dices)) if dices else float('nan')


def _ckpt_path(ckpt_dir: Path, opt_name: str, seed: int) -> Path:
    return ckpt_dir / f"MedSeg_UNet2D_{opt_name}_seed{seed}.pt"


def run_single(opt_name: str, seed: int, epochs: int, batch_size: int, data_root: str, results_dir: Path, resume: bool, ckpt_dir: Path):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr_imgs, tr_masks, va_imgs, va_masks = split_dataset(data_root)
    if len(tr_imgs) == 0:
        print('No dataset found; falling back to synthetic dataset.')
        data_root = synthetic_dataset(n=40, size=128)
        tr_imgs, tr_masks, va_imgs, va_masks = split_dataset(data_root)

    train_ds = SegDataset(tr_imgs, tr_masks, size=256)
    val_ds = SegDataset(va_imgs, va_masks, size=256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = UNet2D().to(device)

    name = opt_name.upper()
    if name in ('ADAM',):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif name in ('SGD', 'SGD_MOMENTUM', 'SGD-MOMENTUM'):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9 if 'MOMENTUM' in name else 0.0)
    else:
        raise ValueError(f'Unsupported optimizer: {opt_name}')

    history = []
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = _ckpt_path(ckpt_dir, opt_name, seed)
    start_epoch = 1
    if resume and ckpt_file.exists():
        try:
            state = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(state['model'], strict=False)
            if isinstance(optimizer, torch.optim.SGD) and state.get('opt', {}).get('type') == 'SGD':
                optimizer.load_state_dict(state['optimizer'])
            if isinstance(optimizer, torch.optim.Adam) and state.get('opt', {}).get('type') == 'Adam':
                optimizer.load_state_dict(state['optimizer'])
            start_epoch = int(state.get('epoch', 0)) + 1
            history = state.get('history', [])
            print(f"Resuming from epoch {start_epoch}: {ckpt_file}")
        except Exception as e:
            print('Resume failed:', e)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_dice = evaluate(model, val_loader, device)
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_dice': val_dice})
        print(f'seed={seed} {opt_name} [{epoch}/{epochs}] loss={tr_loss:.4f} val_dice={val_dice:.3f}')
        try:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'history': history,
                'opt': {'type': 'Adam' if isinstance(optimizer, torch.optim.Adam) else 'SGD'},
                'seed': seed,
            }, ckpt_file)
        except Exception as e:
            print('Warning: failed to save checkpoint:', e)
    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None

    df = pd.DataFrame(history)
    df['elapsed_seconds'] = elapsed
    df['peak_gpu_mb'] = peak_mb
    out = results_dir / f'NN_UNet2D_MedSeg_{opt_name}_seed{seed}_publication.csv'
    df.to_csv(out, index=False)
    return out


def compute_statistics(results_dir: str):
    import glob, re
    patterns = {
        'Adam': f"{results_dir}/NN_UNet2D_MedSeg_Adam_*_publication.csv",
        'SGD_Momentum': f"{results_dir}/NN_UNet2D_MedSeg_SGD_Momentum_*_publication.csv",
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
            vals[seed] = float(df['val_dice'].iloc[-1]) if 'val_dice' in df.columns else float('nan')
        data[opt] = vals
    rows = []
    A, B = 'Adam', 'SGD_Momentum'
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
        out = Path(results_dir) / 'medseg_statistical_comparisons_publication.csv'
        df.to_csv(out, index=False)
        print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description='Medical Segmentation (2D U-Net) Kaggle Publication Suite')
    parser.add_argument('--data-root', type=str, default='')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint if available')
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints_medseg')
    args, _ = parser.parse_known_args()

    if args.quick:
        seeds = [1, 2, 3]
        epochs = 5
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        epochs = args.epochs

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root
    if not data_root:
        # If not provided, try a common Kaggle input mount or fallback to synthetic
        data_root = os.environ.get('DATA_ROOT', '')
    if not data_root:
        print('No data-root provided; script will generate a small synthetic dataset.')

    configs = [
        ('Adam', None),
        ('SGD_Momentum', None),
    ]

    total = len(configs) * len(seeds)
    print(f'Total runs: {total}')

    ckpt_dir = Path(args.ckpt_dir)
    for opt, _ in configs:
        for seed in seeds:
            try:
                run_single(opt, seed, epochs, args.batch_size, data_root, results_dir, args.resume, ckpt_dir)
            except Exception as e:
                print('Error:', e)
    compute_statistics(str(results_dir))


if __name__ == '__main__':
    main()
