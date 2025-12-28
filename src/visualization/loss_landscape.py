import torch
import numpy as np
from typing import Tuple, List

# Matplotlib-based plotting helpers used by the deliverables generator
import matplotlib
# Use a non-interactive backend to make plotting/animation safe in CI and headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import os
import tempfile


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
            ii = int(i); jj = int(j)
            new_vec = base + float(alphas[ii]) * dir1 + float(betas[jj]) * dir2
            _set_params_from_vector(model, new_vec)
            Z[ii, jj] = evaluate_loss(model, loader, criterion, device, max_batches=max_batches)
    _set_params_from_vector(model, base)
    A, B = np.meshgrid(alphas, betas, indexing='ij')
    return A, B, Z


# Visualization helpers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from pathlib import Path
import logging
import json





def plot_loss_landscape(test_function,
                        x_range: Tuple[float, float] = (-2, 2),
                        y_range: Tuple[float, float] = (-2, 2),
                        num_points: int = 100,
                        save_path: str | None = None,
                        cmap: str = 'viridis',
                        contour: bool = True):
    """Create a 2D loss landscape contour (matplotlib).

    Args:
        test_function: callable taking a 2D numpy array-like [x, y] -> scalar
        x_range, y_range: bounds for grid
        num_points: resolution per axis
        save_path: if provided, saves figure to this path and returns the path
        cmap: matplotlib colormap
        contour: whether to use contourf (True) or pcolormesh (False)

    Returns:
        matplotlib.Figure or saved path (str)
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=np.float32)
    for i in range(num_points):
        for j in range(num_points):
            ii = int(i); jj = int(j)
            Z[ii, jj] = float(test_function(np.array([X[ii, jj], Y[ii, jj]])))

    fig, ax = plt.subplots(figsize=(6, 5))
    if contour:
        cs = ax.contourf(X, Y, Z, 50, cmap=cmap)
    else:
        cs = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)

    fig.colorbar(cs, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(getattr(test_function, '__name__', 'loss_landscape'))

    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(save_path)

    return fig


def create_loss_landscape_animation(test_function,
                                    trajectory=None,
                                    x_range: Tuple[float, float] = (-2, 2),
                                    y_range: Tuple[float, float] = (-2, 2),
                                    num_points: int = 100,
                                    n_frames: int = 30,
                                    save_path: str | None = None,
                                    cmap: str = 'viridis',
                                    interval: int = 100,
                                    fps: int | None = None):
    """Create an animation of the loss landscape.

    Two modes are supported:
      - If `trajectory` is provided (array-like shape (T,2)), an animation shows the
        trajectory points moving over a static loss landscape.
      - If `trajectory` is None, a conservative sweep animation is created (suitable
        for CI environments using PillowWriter).

    Args:
        test_function: callable taking a 2D numpy array-like [x, y] -> scalar
        trajectory: optional array-like of shape (T,2) giving points to animate
        x_range, y_range: bounds for grid
        num_points: resolution per axis
        n_frames: number of frames in the sweep mode
        save_path: if provided, saved animated GIF/MP4 path is returned
        interval: ms between frames (used to compute fps for PillowWriter)

    Returns:
        matplotlib.animation.FuncAnimation or saved path (str)
    """
    X_lin = np.linspace(x_range[0], x_range[1], num_points)
    Y_lin = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(X_lin, Y_lin)

    if trajectory is not None:
        # Static landscape, animate trajectory marker
        Z = np.zeros_like(X, dtype=np.float32)
        for i in range(num_points):
            for j in range(num_points):
                Z[i, j] = float(test_function(np.array([X[i, j], Y[i, j]])))

        fig, ax = plt.subplots(figsize=(6, 5))
        cs = ax.contourf(X, Y, Z, levels=60, cmap=cmap)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        traj = np.asarray(trajectory)
        line, = ax.plot([], [], 'o-', color='red', markersize=4)

        def init():
            line.set_data([], [])
            return (line,)

        def update(frame):
            line.set_data(traj[:frame + 1, 0], traj[:frame + 1, 1])
            return (line,)

        anim = animation.FuncAnimation(fig, update, frames=len(traj), init_func=init, blit=True)
    else:
        # Sweep mode (no trajectory) - produce a simple animated variation
        Zs = []
        for t in range(n_frames):
            shift = 0.3 * np.sin(2 * np.pi * t / max(1, n_frames))
            Z = np.zeros_like(X, dtype=np.float32)
            for i in range(num_points):
                for j in range(num_points):
                    ii = int(i); jj = int(j)
                    pt = np.array([X[ii, jj] + shift, Y[ii, jj]])
                    Z[ii, jj] = float(test_function(pt))
            Zs.append(Z)

        fig, ax = plt.subplots(figsize=(6, 5))
        cs = ax.contourf(X, Y, Zs[0], 50, cmap=cmap)
        fig.colorbar(cs, ax=ax)

        def update(frame_idx):
            ax.clear()
            ax.contourf(X, Y, Zs[frame_idx], 50, cmap=cmap)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'frame: {frame_idx}')
            return []

        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        # determine fps to use
        fps_use = fps if (fps is not None) else max(1, int(1000 // max(1, interval)))
        writer = PillowWriter(fps=fps_use)
        anim.save(save_path, writer=writer)
        plt.close(fig)
        return str(save_path)

    return anim
