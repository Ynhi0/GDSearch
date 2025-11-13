"""
Medical image segmentation (3D U-Net, MONAI) with Dice metric logging and CSV output.

This is a scaffold that expects user-provided data dicts or a dataset loader.
Imports MONAI lazily to avoid hard dependency during CI or environments without MONAI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch


def _try_import_monai():
    try:
        from monai.networks.nets import UNet
        from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord
        from monai.data import CacheDataset, DataLoader
        from monai.losses import DiceLoss
        from monai.metrics import DiceMetric
        return UNet, Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, CacheDataset, DataLoader, DiceLoss, DiceMetric
    except Exception as e:
        raise RuntimeError("MONAI is required for this script. Install via `pip install monai[all]`." ) from e


def medical_image_segmentation(data_dicts: List[Dict[str, str]] | None = None, epochs: int = 5, batch_size: int = 2):
    UNet, Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, CacheDataset, DataLoader, DiceLoss, DiceMetric = _try_import_monai()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"]),
    ])

    # Load dataset (placeholder paths if none provided)
    if data_dicts is None:
        data_dicts = [
            {"image": "path/to/image1.nii", "label": "path/to/label1.nii"},
            {"image": "path/to/image2.nii", "label": "path/to/label2.nii"},
            {"image": "path/to/image3.nii", "label": "path/to/label3.nii"},
        ]

    # Simple split: last sample as "validation"
    train_data = data_dicts[:-1]
    val_data = data_dicts[-1:]
    train_ds = CacheDataset(data=train_data, transform=train_transforms)
    val_ds = CacheDataset(data=val_data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Model & loss/metrics
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2), num_res_units=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = DiceLoss(sigmoid=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    history = []
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        # simple validation Dice on val_loader
        model.eval()
        dices = []
        with torch.no_grad():
            for vb in val_loader:
                vimg, vlab = vb['image'].to(device), vb['label'].to(device)
                vlogits = model(vimg)
                vprob = torch.sigmoid(vlogits)
                vbin = (vprob > 0.5).float()
                dice_metric.reset()
                dice_metric(vbin, vlab)
                d = dice_metric.aggregate().item()
                dices.append(d)
        avg_dice = float(np.mean(dices)) if dices else float('nan')

        history.append({'epoch': epoch, 'train_loss': epoch_loss, 'val_dice': avg_dice})
        print(f"Epoch {epoch}/{epochs} - loss={epoch_loss:.4f} val_dice={avg_dice:.4f}")

    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None

    df = pd.DataFrame(history)
    df['elapsed_seconds'] = elapsed
    df['peak_gpu_mb'] = peak_mb
    Path('results').mkdir(parents=True, exist_ok=True)
    out = Path('results') / 'APP_MedicalSeg_UNet_dummy_application.csv'
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    medical_image_segmentation()