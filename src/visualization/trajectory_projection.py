"""
High-Dimensional Trajectory Projection for Neural Network Optimization Visualization.

CRITICAL FIX: Addresses the limitation that trajectory visualization only works
for 2D synthetic functions. This module projects high-dimensional optimization
paths (e.g., 1M+ parameters) onto 2D planes for visualization.

Enables "Optimization Path in Latent Space" plots showing how SGD oscillates
while Adam goes straight—a key visual insight missing from the current codebase.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


class TrajectoryProjector:
    """
    Project high-dimensional parameter trajectories onto 2D for visualization.
    
    Use Cases:
    1. Visualize optimization path of neural networks (millions of parameters)
    2. Compare trajectory "smoothness" between optimizers
    3. Show how different optimizers explore parameter space
    
    Methods:
    - PCA: Preserves global structure, shows main variance directions
    - t-SNE: Preserves local structure, shows clustering
    - Random Projection: Fast baseline, preserves distances approximately
    """
    
    def __init__(
        self,
        method: str = 'pca',
        n_components: int = 2,
        subsample_params: Optional[int] = 10000
    ):
        """
        Initialize trajectory projector.
        
        CRITICAL FIX: Default subsample_params set to 10,000 to prevent OOM crashes.
        
        MEMORY SAFETY ANALYSIS:
        - ResNet-18: ~11M parameters × 4 bytes (float32) = 44 MB per snapshot
        - Training for 100 epochs with snapshot_every=1 → 100 snapshots → 4.4 GB
        - t-SNE requires O(n²) pairwise distances → memory explosion for large models
        
        With subsample_params=10000 (default):
        - 10K params × 4 bytes × 100 snapshots = 4 MB (safe for all systems)
        - Sufficient dimensionality for meaningful trajectory visualization
        - Preserves trajectory structure via random sampling
        
        CRITICAL: Users must EXPLICITLY set subsample_params=None to use full
        parameter space, acknowledging the memory risk.
        
        Args:
            method: Projection method ('pca', 'tsne', 'random')
            n_components: Number of dimensions to project to (typically 2)
            subsample_params: Number of parameters to randomly sample (DEFAULT: 10000).
                            Set to None ONLY for small models (<100K params) with
                            sufficient RAM (32GB+). Prevents accidental OOM crashes.
        """
        self.method = method
        self.n_components = n_components
        self.subsample_params = subsample_params
        self.projector = None
        self.param_indices = None  # For subsampling
        
    def collect_trajectory(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_steps: int,
        device: torch.device,
        snapshot_every: int = 10
    ) -> List[np.ndarray]:
        """
        Collect parameter snapshots during training.
        
        Args:
            model: Neural network
            optimizer: Optimizer
            train_loader: Training data
            criterion: Loss function
            num_steps: Number of training steps
            device: Computation device
            snapshot_every: Save parameters every N steps
            
        Returns:
            List of parameter snapshots (each is a flattened numpy array)
        """
        model.train()
        snapshots = []
        
        # Get initial parameters
        snapshots.append(self._extract_params(model))
        
        dataiter = iter(train_loader)
        for step in range(num_steps):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(train_loader)
                batch = next(dataiter)
            
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
            
            # Training step
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Save snapshot
            if (step + 1) % snapshot_every == 0:
                snapshots.append(self._extract_params(model))
        
        logging.info(f"Collected {len(snapshots)} parameter snapshots over {num_steps} steps")
        return snapshots
    
    def _extract_params(self, model: nn.Module) -> np.ndarray:
        """
        Extract model parameters as flattened numpy array.
        
        Optionally subsample parameters to reduce memory.
        """
        # Flatten all parameters into single vector
        param_vector = torch.cat([p.data.flatten() for p in model.parameters()]).cpu().numpy()
        
        # Subsample if requested
        if self.subsample_params is not None:
            if self.param_indices is None:
                # First call: create random indices
                total_params = len(param_vector)
                if self.subsample_params < total_params:
                    self.param_indices = np.random.choice(
                        total_params, self.subsample_params, replace=False
                    )
                    logging.info(f"Subsampling {self.subsample_params}/{total_params} parameters")
                else:
                    self.param_indices = np.arange(total_params)
            
            param_vector = param_vector[self.param_indices]
        
        return param_vector
    
    def fit_projection(self, trajectories: Dict[str, List[np.ndarray]]) -> None:
        """
        Fit projection on collected trajectories from multiple optimizers.
        
        CRITICAL FIX: To avoid bias toward optimizers with more snapshots,
        we balance the dataset by sampling EQUAL number of snapshots from each
        optimizer trajectory before fitting PCA.
        
        Scientific Justification:
        - If Optimizer A takes 100 steps and Optimizer B takes 500 steps,
          naive concatenation gives B 5x more influence on PCA axes
        - This makes PCA components represent "where B goes" rather than
          "directions of maximum variance across all optimizers equally"
        - Balanced sampling ensures fair comparison
        
        Args:
            trajectories: Dict mapping optimizer names to parameter snapshot lists
        """
        # Find minimum trajectory length to ensure balanced sampling
        min_length = min(len(snapshots) for snapshots in trajectories.values())
        
        if min_length < 2:
            logging.warning(f"Shortest trajectory has only {min_length} snapshots. "
                          "Consider collecting more snapshots for better projection.")
        
        # Balanced sampling: take same number of snapshots from each optimizer
        all_snapshots = []
        for opt_name, snapshots in trajectories.items():
            # Uniformly sample min_length snapshots from this trajectory
            if len(snapshots) > min_length:
                indices = np.linspace(0, len(snapshots) - 1, min_length, dtype=int)
                sampled_snapshots = [snapshots[i] for i in indices]
            else:
                sampled_snapshots = snapshots
            
            all_snapshots.extend(sampled_snapshots)
            logging.debug(f"Sampled {len(sampled_snapshots)} snapshots from {opt_name} "
                        f"(original: {len(snapshots)})")
        
        X = np.array(all_snapshots)
        
        logging.info(f"Fitting {self.method} projection on {X.shape[0]} BALANCED snapshots "
                    f"({min_length} per optimizer × {len(trajectories)} optimizers) "
                    f"with {X.shape[1]} parameters each")
        
        if self.method == 'pca':
            self.projector = PCA(n_components=self.n_components, random_state=42)
            self.projector.fit(X)
            
            # Log explained variance
            explained_var = self.projector.explained_variance_ratio_
            logging.info(f"PCA explained variance: {explained_var[:self.n_components]}")
            logging.info(f"Total variance captured: {np.sum(explained_var):.3f}")
            
        elif self.method == 'tsne':
            # t-SNE doesn't have separate fit/transform, so we store parameters
            self.projector = TSNE(
                n_components=self.n_components,
                random_state=42,
                perplexity=min(30, len(X) - 1)  # Adjust perplexity for small datasets
            )
            logging.warning("t-SNE requires full dataset for fitting, may be slow")
            
        elif self.method == 'random':
            # Random projection matrix
            n_features = X.shape[1]
            self.projector = np.random.randn(n_features, self.n_components)
            self.projector /= np.linalg.norm(self.projector, axis=0)  # Normalize columns
            
        else:
            raise ValueError(f"Unknown projection method: {self.method}")
    
    def project_trajectory(self, snapshots: List[np.ndarray]) -> np.ndarray:
        """
        Project parameter trajectory to low-dimensional space.
        
        CRITICAL FIX FOR t-SNE: t-SNE cannot project new data points because it
        is a manifold learning method that computes embeddings globally. Each
        call to fit_transform creates a completely different embedding space.
        
        For trajectory projection, we MUST use methods that support out-of-sample
        projection (PCA, random projection). t-SNE is disabled for individual
        trajectory projection.
        
        Args:
            snapshots: List of parameter vectors
            
        Returns:
            Array of shape (n_snapshots, n_components)
        """
        if self.projector is None:
            raise RuntimeError("Must call fit_projection() before project_trajectory()")
        
        X = np.array(snapshots)
        
        if self.method == 'pca':
            return self.projector.transform(X)
        elif self.method == 'tsne':
            # CRITICAL: t-SNE cannot project new points consistently
            # Each fit_transform creates a DIFFERENT embedding space
            # Solution: Only allow t-SNE for visualize_trajectories where all
            # data is projected together in one consistent space
            raise RuntimeError(
                "t-SNE cannot project individual trajectories consistently. "
                "Use method='pca' or 'random' for individual trajectory projection, "
                "or use visualize_trajectories() to project all trajectories together."
            )
        elif self.method == 'random':
            return X @ self.projector
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def visualize_trajectories(
        self,
        trajectories: Dict[str, List[np.ndarray]],
        output_path: str,
        title: str = "Optimizer Trajectories in Parameter Space"
    ) -> None:
        """
        Create publication-quality trajectory visualization.
        
        CRITICAL FIX FOR t-SNE: To use t-SNE, ALL trajectories must be projected
        together in a single fit_transform call. This ensures all points exist
        in the same embedding space, making comparisons meaningful.
        
        Args:
            trajectories: Dict mapping optimizer names to snapshot lists
            output_path: Path to save figure
            title: Plot title
        """
        # For t-SNE, we must project all data together
        if self.method == 'tsne':
            # Concatenate all snapshots from all trajectories
            all_snapshots = []
            trajectory_lengths = {}
            
            for opt_name, snapshots in trajectories.items():
                trajectory_lengths[opt_name] = len(snapshots)
                all_snapshots.extend(snapshots)
            
            # Project everything together in ONE consistent space
            X_all = np.array(all_snapshots)
            projected_all = self.projector.fit_transform(X_all)
            
            # Split back into individual trajectories
            projected_trajectories = {}
            offset = 0
            for opt_name, length in trajectory_lengths.items():
                projected_trajectories[opt_name] = projected_all[offset:offset+length]
                offset += length
        else:
            # For PCA and random projection, fit once then transform each trajectory
            if self.projector is None:
                self.fit_projection(trajectories)
            
            # Project each trajectory separately (they share the same projection)
            projected_trajectories = {}
            for opt_name, snapshots in trajectories.items():
                projected = self.project_trajectory(snapshots)
                projected_trajectories[opt_name] = projected
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        colors = sns.color_palette("husl", len(projected_trajectories))
        
        for (opt_name, trajectory), color in zip(projected_trajectories.items(), colors):
            # Plot trajectory as connected line
            plt.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                'o-',
                color=color,
                alpha=0.7,
                linewidth=2,
                markersize=4,
                label=opt_name
            )
            
            # Mark start point
            plt.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                marker='*',
                s=300,
                color=color,
                edgecolors='black',
                linewidths=2,
                zorder=10
            )
            
            # Mark end point
            plt.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                marker='X',
                s=200,
                color=color,
                edgecolors='black',
                linewidths=2,
                zorder=10
            )
        
        plt.xlabel(f'{self.method.upper()} Component 1', fontsize=14)
        plt.ylabel(f'{self.method.upper()} Component 2', fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        plt.text(
            0.02, 0.98,
            '★ Start  ✕ End',
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved trajectory visualization to {output_path}")
    
    def analyze_trajectory_smoothness(
        self,
        snapshots: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Quantify trajectory smoothness/oscillation.
        
        Args:
            snapshots: Parameter trajectory
            
        Returns:
            Dict with smoothness metrics
        """
        trajectory = np.array(snapshots)
        
        # Compute step sizes (consecutive distances)
        step_sizes = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        
        # Compute direction changes (angle between consecutive steps)
        direction_changes = []
        for i in range(len(trajectory) - 2):
            v1 = trajectory[i+1] - trajectory[i]
            v2 = trajectory[i+2] - trajectory[i+1]
            
            # Cosine similarity
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-12 and v2_norm > 1e-12:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                direction_changes.append(angle)
        
        # Total path length
        total_length = np.sum(step_sizes)
        
        # Euclidean distance (start to end)
        euclidean_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        
        # Path efficiency: ratio of direct distance to path length
        path_efficiency = euclidean_distance / (total_length + 1e-12)
        
        return {
            'total_path_length': float(total_length),
            'euclidean_distance': float(euclidean_distance),
            'path_efficiency': float(path_efficiency),
            'mean_step_size': float(np.mean(step_sizes)),
            'std_step_size': float(np.std(step_sizes)),
            'mean_direction_change': float(np.mean(direction_changes)) if direction_changes else 0.0,
            'std_direction_change': float(np.std(direction_changes)) if direction_changes else 0.0,
        }
