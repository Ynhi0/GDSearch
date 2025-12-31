"""
Strict data splitting utilities to prevent data leakage and adaptive overfitting.

This module enforces the "Two-Stage Protocol" mandated for rigorous research standards:
    1. Hyperparameter Tuning: Use ONLY train + validation splits
    2. Final Evaluation: Use ONLY test split with frozen hyperparameters

References:
- Cawley & Talbot (JMLR 2010): "On Over-fitting in Model Selection"
- Raschka (2018): "Model Evaluation, Model Selection, and Algorithm Selection"
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from typing import Tuple, Optional, Dict
import logging


class DataSplitManager:
    """
    Manages strict train/val/test splits with data leakage prevention.
    
    Key features:
    - Immutable test set (isolated from tuning)
    - Reproducible splits via fixed seeds
    - Stratification support for classification
    - Validation that test set is never accessed during tuning
    """
    
    def __init__(self, dataset: Dataset, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42,
                 stratify_labels: Optional[np.ndarray] = None):
        """
        Initialize data split manager.
        
        Args:
            dataset: PyTorch Dataset object
            train_ratio: Fraction for training (default: 0.7)
            val_ratio: Fraction for validation/tuning (default: 0.15)
            test_ratio: Fraction for final testing (default: 0.15)
            seed: Random seed for reproducibility
            stratify_labels: Optional labels for stratified splitting
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Track if test set has been accessed (for validation)
        self._test_accessed = False
        self._hyperparams_frozen = False
        
        # Perform split
        self.train_indices, self.val_indices, self.test_indices = self._create_splits(stratify_labels)
        
        logging.info(f"Data splits created: Train={len(self.train_indices)}, "
                    f"Val={len(self.val_indices)}, Test={len(self.test_indices)}")
    
    def _create_splits(self, stratify_labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create reproducible train/val/test splits."""
        from src.utils.safe_len import len_sized
        n_total = len_sized(self.dataset)
        indices = np.arange(int(n_total))
        
        # Set seed for reproducibility
        rng = np.random.default_rng(self.seed)
        
        if stratify_labels is not None:
            # Stratified split (maintains class distribution)
            train_idx, val_idx, test_idx = self._stratified_split(indices, stratify_labels, rng)
        else:
            # Random split
            rng.shuffle(indices)
            
            train_size = int(n_total * self.train_ratio)
            val_size = int(n_total * self.val_ratio)
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
        
        return train_idx, val_idx, test_idx
    
    def _stratified_split(self, indices: np.ndarray, labels: np.ndarray, 
                         rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform stratified split to maintain class distribution."""
        unique_labels = np.unique(labels)
        train_idx_list, val_idx_list, test_idx_list = [], [], []
        
        for label in unique_labels:
            # Get indices for this class
            class_indices = indices[labels == label]
            rng.shuffle(class_indices)
            
            n_class = len(class_indices)
            train_size = int(n_class * self.train_ratio)
            val_size = int(n_class * self.val_ratio)
            
            train_idx_list.append(class_indices[:train_size])
            val_idx_list.append(class_indices[train_size:train_size + val_size])
            test_idx_list.append(class_indices[train_size + val_size:])
        
        return (np.concatenate(train_idx_list),
                np.concatenate(val_idx_list),
                np.concatenate(test_idx_list))
    
    def get_train_loader(self, batch_size: int = 32, **kwargs) -> DataLoader:
        """Get DataLoader for training set."""
        train_subset = Subset(self.dataset, self.train_indices.tolist())
        return DataLoader(train_subset, batch_size=batch_size, shuffle=True, **kwargs)
    
    def get_val_loader(self, batch_size: int = 32, **kwargs) -> DataLoader:
        """
        Get DataLoader for validation set (used during hyperparameter tuning).
        
        IMPORTANT: This should be the ONLY set used for model selection.
        """
        if self._hyperparams_frozen:
            logging.warning("Accessing validation set after hyperparameters are frozen. "
                          "This may indicate improper protocol!")
        
        val_subset = Subset(self.dataset, self.val_indices.tolist())
        return DataLoader(val_subset, batch_size=batch_size, shuffle=False, **kwargs)
    
    def get_test_loader(self, batch_size: int = 32, **kwargs) -> DataLoader:
        """
        Get DataLoader for test set (used ONLY for final evaluation).
        
        CRITICAL: This must not be accessed during hyperparameter tuning.
        This method will *raise* if hyperparameters are not frozen to prevent
        accidental data leakage (fail-fast behavior).
        """
        if not self._hyperparams_frozen:
            # Fail fast: raise an exception to prevent using the test set during tuning
            raise RuntimeError(
                "PROTOCOL VIOLATION: Attempted to access TEST set before hyperparameters were frozen. "
                "Call DataSplitManager.freeze_hyperparameters(best_hyperparams) after tuning before requesting the test loader. "
                "Accessing the test set during tuning invalidates results."
            )

        self._test_accessed = True
        test_subset = Subset(self.dataset, self.test_indices.tolist())
        return DataLoader(test_subset, batch_size=batch_size, shuffle=False, **kwargs)
    
    def freeze_hyperparameters(self, best_hyperparams: Dict):
        """
        Freeze hyperparameters after tuning phase.
        
        This signals the transition from Stage 1 (Tuning) to Stage 2 (Evaluation).
        After calling this, test set can be accessed for final evaluation.
        
        Args:
            best_hyperparams: Dictionary of selected hyperparameters
        """
        self._hyperparams_frozen = True
        self.best_hyperparams = best_hyperparams
        
        logging.info("="*70)
        logging.info("HYPERPARAMETERS FROZEN - Transitioning to Final Evaluation")
        logging.info("="*70)
        logging.info(f"Selected hyperparameters: {best_hyperparams}")
        logging.info("Test set can now be accessed for final unbiased evaluation.")
        logging.info("="*70)
    
    def validate_protocol(self) -> bool:
        """
        Validate that proper experimental protocol was followed.
        
        Returns:
            True if protocol was followed correctly, False otherwise
        """
        if self._test_accessed and not self._hyperparams_frozen:
            logging.error("PROTOCOL VIOLATION DETECTED!")
            logging.error("   Test set was accessed without freezing hyperparameters first.")
            return False
        
        if not self._test_accessed:
            logging.warning("Test set was never accessed. Results may be incomplete.")
            return True
        
        if self._hyperparams_frozen and self._test_accessed:
            logging.info("Proper protocol followed:")
            logging.info("   1. Hyperparameters tuned on train+val")
            logging.info("   2. Hyperparameters frozen")
            logging.info("   3. Final evaluation on test set")
            return True
        
        return True
    
    def get_split_info(self) -> Dict:
        """Get information about the data splits."""
        from src.utils.safe_len import len_sized
        return {
            'total_size': len_sized(self.dataset),
            'train_size': len(self.train_indices),
            'val_size': len(self.val_indices),
            'test_size': len(self.test_indices),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'seed': self.seed,
            'test_accessed': self._test_accessed,
            'hyperparams_frozen': self._hyperparams_frozen
        }


def create_nested_cv_splits(dataset: Dataset, 
                           n_outer_folds: int = 5,
                           n_inner_folds: int = 3,
                           seed: int = 42) -> list:
    """
    Create nested cross-validation splits for unbiased model selection.
    
    Nested CV structure:
    - Outer loop: For unbiased performance estimation
    - Inner loop: For hyperparameter tuning
    
    This is the GOLD STANDARD for model selection when data is limited,
    as recommended by Cawley & Talbot (2010).
    
    Args:
        dataset: PyTorch Dataset
        n_outer_folds: Number of outer CV folds (default: 5)
        n_inner_folds: Number of inner CV folds (default: 3)
        seed: Random seed
        
    Returns:
        List of (outer_train_idx, outer_test_idx, inner_splits) tuples
    """
    from sklearn.model_selection import KFold
    
    from src.utils.safe_len import len_sized
    n_samples = len_sized(dataset)
    indices = np.arange(int(n_samples))
    
    outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    nested_splits = []
    
    for outer_train_idx, outer_test_idx in outer_cv.split(indices):
        # Inner CV for hyperparameter tuning
        inner_cv = KFold(n_splits=n_inner_folds, shuffle=True, random_state=seed + 1)
        inner_splits = list(inner_cv.split(outer_train_idx))
        
        nested_splits.append({
            'outer_train': outer_train_idx,
            'outer_test': outer_test_idx,
            'inner_splits': inner_splits
        })
    
    logging.info(f"Created nested CV: {n_outer_folds} outer folds × {n_inner_folds} inner folds")
    return nested_splits


class HyperparameterTuningGuard:
    """
    Context manager to enforce proper tuning/evaluation separation.
    
    Usage:
        with HyperparameterTuningGuard(split_manager):
            # Tuning phase - can access train and val
            # CANNOT access test
            optimizer = tune_hyperparameters(train_loader, val_loader)
        
        # After context exit, hyperparameters are automatically frozen
        final_results = evaluate(test_loader)
    """
    
    def __init__(self, split_manager: DataSplitManager):
        self.split_manager = split_manager
        self.best_hyperparams = None
    
    def __enter__(self):
        logging.info("Entering hyperparameter tuning phase...")
        logging.info("Test set access is BLOCKED during this phase.")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.best_hyperparams is None:
            logging.warning("No hyperparameters were set during tuning phase!")
        else:
            self.split_manager.freeze_hyperparameters(self.best_hyperparams)
        
        logging.info("Exiting hyperparameter tuning phase.")
    
    def set_best_hyperparams(self, hyperparams: Dict):
        """Set the best hyperparameters found during tuning."""
        self.best_hyperparams = hyperparams
        logging.info(f"Best hyperparameters selected: {hyperparams}")


# Utility function for backwards compatibility
def get_train_val_test_loaders(dataset: Dataset,
                               batch_size: int = 32,
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15,
                               seed: int = 42,
                               **loader_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to get train/val/test loaders with proper protocol.
    
    NOTE: This returns loaders directly. For strict protocol enforcement,
    use DataSplitManager class instead.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size for loaders
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed
        **loader_kwargs: Additional DataLoader arguments
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    manager = DataSplitManager(dataset, train_ratio, val_ratio, test_ratio, seed)
    
    train_loader = manager.get_train_loader(batch_size, **loader_kwargs)
    val_loader = manager.get_val_loader(batch_size, **loader_kwargs)
    
    # Issue warning since we're bypassing protocol
    logging.warning("Using convenience function bypasses protocol validation.")
    logging.warning("   Consider using DataSplitManager for strict enforcement.")
    
    test_loader = manager.get_test_loader(batch_size, **loader_kwargs)
    
    return train_loader, val_loader, test_loader
