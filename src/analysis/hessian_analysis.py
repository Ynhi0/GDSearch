"""
Hessian Spectrum Analysis for validating optimizer quality.

This module provides tools to analyze the loss landscape curvature and
flatness of minima found by optimizers - critical evidence for rigorous research.

Key analyses:
- Hessian eigenvalue spectrum (λ_min, λ_max, condition number)
- Flatness measures (trace, Frobenius norm)
- Sharpness-Aware Minimization (SAM) validation
- Filter-normalized loss surface visualization

References:
- Keskar et al. (2017): "On Large-Batch Training for Deep Learning"
- Foret et al. (2021): "Sharpness-Aware Minimization"
- Li et al. (2018): "Visualizing the Loss Landscape of Neural Nets"
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import logging
import matplotlib.pyplot as plt
from pathlib import Path


class HessianAnalyzer:
    """
    Compute and analyze Hessian spectrum for neural networks.
    
    Provides evidence of:
    - Flatness of minima (better generalization)
    - Optimizer quality (SAM should find flatter minima than SGD)
    - Conditioning of optimization landscape
    """
    
    def __init__(self, model: nn.Module, criterion: nn.Module, device: torch.device = None):
        """
        Initialize Hessian analyzer.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            device: Computation device
        """
        self.model = model
        self.criterion = criterion
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def compute_hessian_eigenvalues(self, 
                                    dataloader: DataLoader,
                                    num_batches: int = 10,
                                    top_k: int = 10) -> Dict[str, float]:
        """
        Compute top eigenvalues of Hessian matrix.
        
        Uses power iteration method for efficiency (full Hessian is O(n²) memory).
        
        Args:
            dataloader: DataLoader for computing Hessian
            num_batches: Number of batches to use (more = more accurate)
            top_k: Number of top eigenvalues to compute
            
        Returns:
            Dictionary with:
            - eigenvalues: Top k eigenvalues (sorted descending)
            - max_eigenvalue: Largest eigenvalue (sharpness indicator)
            - trace_estimate: Estimated trace (sum of all eigenvalues)
        """
        logging.info(f"Computing top {top_k} Hessian eigenvalues...")
        
        # Get model parameters as flat vector
        params = [p for p in self.model.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in params)
        
        logging.info(f"Total parameters: {num_params:,}")
        
        if num_params > 100000:
            logging.warning(f"Large model ({num_params:,} params) - Hessian computation may be slow")
        
        # Compute Hessian eigenvalues using power iteration
        eigenvalues = self._power_iteration_eigenvalues(dataloader, params, num_batches, top_k)
        
        return {
            'eigenvalues': eigenvalues,
            'max_eigenvalue': float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0,
            'min_eigenvalue': float(eigenvalues[-1]) if len(eigenvalues) > 0 else 0.0,
            'condition_number': float(eigenvalues[0] / eigenvalues[-1]) if len(eigenvalues) > 1 and eigenvalues[-1] != 0 else float('inf'),
            'trace_estimate': float(eigenvalues.sum())
        }
    
    def _power_iteration_eigenvalues(self, 
                                     dataloader: DataLoader,
                                     params: List[torch.Tensor],
                                     num_batches: int,
                                     top_k: int) -> torch.Tensor:
        """
        Compute top eigenvalues using power iteration (Lanczos method).
        
        This is much more efficient than computing full Hessian for large models.
        """
        batch_iter = iter(dataloader)
        
        # Hessian-vector product function
        def hvp(vector):
            """Compute Hessian-vector product H @ v"""
            # Zero gradients
            self.model.zero_grad()
            
            # Accumulate over multiple batches for stability
            hv = None
            for _ in range(num_batches):
                try:
                    inputs, targets = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(dataloader)
                    inputs, targets = next(batch_iter)
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Compute gradients
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Compute gradient
                grads = torch.autograd.grad(loss, params, create_graph=True)
                
                # Compute directional derivative (g^T @ v)
                grad_vector = torch.cat([g.view(-1) for g in grads])
                dot = torch.dot(grad_vector, vector)
                
                # Compute Hessian-vector product
                hv_batch = torch.autograd.grad(dot, params, retain_graph=False)
                hv_batch = torch.cat([h.view(-1) for h in hv_batch])
                
                if hv is None:
                    hv = hv_batch
                else:
                    hv = hv + hv_batch
            
            return hv / num_batches
        
        # Power iteration to find top eigenvalue/eigenvector
        num_params = sum(p.numel() for p in params)
        
        # Initialize random vector
        v = torch.randn(num_params, device=self.device)
        v = v / torch.norm(v)
        
        eigenvalues = []
        eigenvectors = []
        
        # Compute top k eigenvalues using deflation
        # CRITICAL FIX (HIGH-1): Proper deflation for multi-eigenvalue estimation
        # Without deflation, all iterations converge to the top eigenvalue
        for k in range(min(top_k, 10)):  # Limit to 10 for efficiency
            # Power iteration
            for iteration in range(30):  # Increased from 20 for better convergence
                v_new = hvp(v)
                
                # Deflate: subtract projection onto previously found eigenvectors
                for prev_eigenval, prev_eigenvec in zip(eigenvalues, eigenvectors):
                    projection = torch.dot(v_new, prev_eigenvec)
                    v_new = v_new - projection * prev_eigenvec
                
                # Normalize
                norm = torch.norm(v_new)
                if norm < 1e-10:
                    # Orthogonalization collapsed the vector - no more distinct eigenvalues
                    logging.warning(f"Eigenvalue computation stopped at k={k} due to numerical collapse")
                    break
                
                v = v_new / norm
            else:
                # Converged successfully
                v_normalized = v / torch.norm(v)
                Hv = hvp(v_normalized)
                eigenvalue = torch.dot(Hv, v_normalized)
                
                eigenvalues.append(eigenvalue.item())
                eigenvectors.append(v_normalized.clone())
                
                logging.debug(f"Eigenvalue {k+1}: {eigenvalue.item():.6f}")
        
        if len(eigenvalues) == 0:
            logging.warning("No eigenvalues computed - returning zero")
            return torch.tensor([0.0])
        
        return torch.tensor(eigenvalues)
    
    def compute_sharpness(self, dataloader: DataLoader, rho: float = 0.05) -> float:
        """
        Compute SAM-style sharpness metric.
        
        Sharpness = max_{||δ|| ≤ ρ} L(θ + δ) - L(θ)
        
        This measures the maximum loss increase in a small neighborhood,
        which correlates with generalization performance.
        
        Args:
            dataloader: DataLoader for computing loss
            rho: Neighborhood radius (default: 0.05)
            
        Returns:
            Sharpness value (lower is better)
        """
        logging.info(f"Computing sharpness (SAM metric with ρ={rho})...")
        
        # Compute base loss
        base_loss = self._compute_loss(dataloader)
        
        # Compute gradient
        self.model.zero_grad()
        batch_iter = iter(dataloader)
        inputs, targets = next(batch_iter)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        
        # Compute adversarial perturbation
        with torch.no_grad():
            # Gradient norm
            grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in self.model.parameters() if p.grad is not None))
            
            # Perturbation: ε = ρ * grad / ||grad||
            scale = rho / (grad_norm + 1e-12)
            
            # Apply perturbation
            for p in self.model.parameters():
                if p.grad is not None:
                    p.data.add_(p.grad * scale)
        
        # Compute perturbed loss
        perturbed_loss = self._compute_loss(dataloader)
        
        # Restore original parameters
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.data.sub_(p.grad * scale)
        
        sharpness = perturbed_loss - base_loss
        logging.info(f"Sharpness: {sharpness:.6f} (lower is better)")
        
        return float(sharpness)
    
    def _compute_loss(self, dataloader: DataLoader, num_batches: int = 10) -> float:
        """Compute average loss over dataloader."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                count += 1
        
        self.model.train()
        return total_loss / count if count > 0 else 0.0
    
    def analyze_optimizer_quality(self, 
                                  dataloader: DataLoader,
                                  optimizer_name: str = "Optimizer") -> Dict[str, Any]:
        """
        Comprehensive analysis of optimizer quality.
        
        Returns metrics that correlate with generalization:
        - Hessian spectrum (top eigenvalues)
        - Sharpness (SAM metric)
        - Conditioning (λ_max / λ_min)
        
        Args:
            dataloader: DataLoader for analysis
            optimizer_name: Name for logging
            
        Returns:
            Dictionary with all metrics
        """
        logging.info(f"\n{'='*70}")
        logging.info(f"Analyzing Optimizer Quality: {optimizer_name}")
        logging.info(f"{'='*70}")
        
        results = {}
        
        # 1. Hessian eigenvalues
        try:
            hessian_results = self.compute_hessian_eigenvalues(dataloader, num_batches=5, top_k=5)
            results.update(hessian_results)
            
            logging.info(f"\nHessian Spectrum:")
            logging.info(f"  Max eigenvalue (λ_max): {hessian_results['max_eigenvalue']:.6f}")
            logging.info(f"  Min eigenvalue (λ_min): {hessian_results['min_eigenvalue']:.6f}")
            logging.info(f"  Condition number: {hessian_results['condition_number']:.2f}")
        except Exception as e:
            logging.warning(f"Failed to compute Hessian eigenvalues: {e}")
        
        # 2. Sharpness
        try:
            sharpness = self.compute_sharpness(dataloader, rho=0.05)
            results['sharpness'] = sharpness
        except Exception as e:
            logging.warning(f"Failed to compute sharpness: {e}")
        
        logging.info(f"{'='*70}\n")
        
        return results


def plot_hessian_spectrum(eigenvalues: np.ndarray,
                          optimizer_names: List[str],
                          save_path: Optional[Path] = None):
    """
    Plot Hessian eigenvalue spectrum for multiple optimizers.
    
    Args:
        eigenvalues: 2D array (n_optimizers, n_eigenvalues)
        optimizer_names: List of optimizer names
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Top eigenvalues
    ax = axes[0]
    for i, (eigs, name) in enumerate(zip(eigenvalues, optimizer_names)):
        ax.plot(range(1, len(eigs) + 1), eigs, 'o-', label=name, markersize=8)
    
    ax.set_xlabel('Eigenvalue Rank', fontsize=12)
    ax.set_ylabel('Eigenvalue Magnitude', fontsize=12)
    ax.set_title('Top Hessian Eigenvalues', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Max eigenvalue comparison (sharpness proxy)
    ax = axes[1]
    max_eigs = [eigs[0] for eigs in eigenvalues]
    colors = [f'C{i}' for i in range(len(optimizer_names))]
    ax.bar(optimizer_names, max_eigs, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Max Eigenvalue (λ_max)', fontsize=12)
    ax.set_title('Sharpness Comparison (Lower is Better)', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Hessian spectrum plot saved to {save_path}")
    
    plt.show()
    return fig


def plot_sharpness_comparison(sharpness_values: List[float],
                              optimizer_names: List[str],
                              save_path: Optional[Path] = None):
    """
    Plot SAM sharpness metric comparison.
    
    Args:
        sharpness_values: List of sharpness values
        optimizer_names: List of optimizer names
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [f'C{i}' for i in range(len(optimizer_names))]
    bars = ax.bar(optimizer_names, sharpness_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Highlight best (lowest sharpness)
    best_idx = np.argmin(sharpness_values)
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.9)
    
    ax.set_ylabel('Sharpness (SAM metric)', fontsize=12)
    ax.set_title('Flatness of Minima (Lower is Better)', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Annotate best
    ax.annotate('Best (Flattest)', 
                xy=(best_idx, sharpness_values[best_idx]),
                xytext=(best_idx, sharpness_values[best_idx] * 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=12, weight='bold', color='green',
                ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Sharpness comparison plot saved to {save_path}")
    
    plt.show()
    return fig


# Example usage
if __name__ == '__main__':
    # Demonstrate usage
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset
    
    # Create dummy model and data
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )
    
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    criterion = nn.CrossEntropyLoss()
    
    # Analyze
    analyzer = HessianAnalyzer(model, criterion)
    results = analyzer.analyze_optimizer_quality(dataloader, optimizer_name="SGD")
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")
