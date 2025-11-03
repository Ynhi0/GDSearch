import torch
import numpy as np
from typing import Tuple, List


def _flatten_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def _set_params_from_vector(model: torch.nn.Module, vec: torch.Tensor):
    """Set model parameters from a 1D tensor (same order as named parameters)."""
    idx = 0
    for p in model.parameters():
        num = p.numel()
        p.data.copy_(vec[idx:idx + num].view_as(p))
        idx += num


def _get_params_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def _random_direction_like(model: torch.nn.Module, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    vec = _get_params_vector(model).cpu()
    v = torch.randn(vec.shape, generator=g, dtype=vec.dtype)
    v /= (v.norm() + 1e-12)
    return v


def evaluate_loss(model: torch.nn.Module, loader, criterion, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            if i >= max_batches:
                break
    return total_loss / max(1, total_n)


def probe_loss_1d(model: torch.nn.Module,
                   loader,
                   criterion,
                   device: torch.device,
                   direction: torch.Tensor,
                   alphas: np.ndarray,
                   max_batches: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Return alphas and losses along a single direction around current params."""
    base = _get_params_vector(model).clone()
    losses = []
    for a in alphas:
        new_vec = base + float(a) * direction
        _set_params_from_vector(model, new_vec)
        losses.append(evaluate_loss(model, loader, criterion, device, max_batches=max_batches))
    # restore
    _set_params_from_vector(model, base)
    return alphas, np.array(losses)


def probe_loss_2d(model: torch.nn.Module,
                   loader,
                   criterion,
                   device: torch.device,
                   dir1: torch.Tensor,
                   dir2: torch.Tensor,
                   alphas: np.ndarray,
                   betas: np.ndarray,
                   max_batches: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return meshgrid (A,B) and loss values Z over 2D directions around current params."""
    base = _get_params_vector(model).clone()
    Z = np.zeros((len(alphas), len(betas)), dtype=np.float32)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            new_vec = base + float(a) * dir1 + float(b) * dir2
            _set_params_from_vector(model, new_vec)
            Z[i, j] = evaluate_loss(model, loader, criterion, device, max_batches=max_batches)
    _set_params_from_vector(model, base)
    A, B = np.meshgrid(alphas, betas, indexing='ij')
    return A, B, Z
