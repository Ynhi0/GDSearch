#!/usr/bin/env python3
"""
Hyperparameter Tuning Result Cache

Saves and loads tuned hyperparameters to avoid redundant Optuna studies.
Each optimizer's best params are cached per dataset/model combination.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TuningCache:
    """Manages cached hyperparameter tuning results."""
    
    def __init__(self, cache_dir: Path):
        """
        Initialize tuning cache.
        
        Args:
            cache_dir: Directory to store cached tuning results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_key(self, dataset: str, model: str, optimizer: str) -> str:
        """Generate cache file name for a specific configuration."""
        return f"{dataset}_{model}_{optimizer}_tuned.json"
    
    def save_tuned_params(
        self,
        dataset: str,
        model: str,
        optimizer: str,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save tuned hyperparameters to cache.
        
        Args:
            dataset: Dataset name (e.g., 'MNIST')
            model: Model name (e.g., 'SimpleMLP')
            optimizer: Optimizer name (e.g., 'SGD')
            params: Best hyperparameters found
            metadata: Optional metadata (best_val_acc, n_trials, etc.)
            
        Returns:
            Path to saved cache file
        """
        cache_key = self.get_cache_key(dataset, model, optimizer)
        cache_file = self.cache_dir / cache_key
        
        cache_data = {
            "dataset": dataset,
            "model": model,
            "optimizer": optimizer,
            "best_params": params,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
            
        logger.info(f"✅ Saved tuning results to cache: {cache_file.name}")
        return cache_file
    
    def load_tuned_params(
        self,
        dataset: str,
        model: str,
        optimizer: str,
        max_age_hours: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load cached hyperparameters if they exist.
        
        Args:
            dataset: Dataset name
            model: Model name
            optimizer: Optimizer name
            max_age_hours: Optional max age in hours (None = no age limit)
            
        Returns:
            Cached params dict or None if not found/expired
        """
        cache_key = self.get_cache_key(dataset, model, optimizer)
        cache_file = self.cache_dir / cache_key
        
        if not cache_file.exists():
            logger.info(f"⏳ No cached tuning found for {optimizer} (first run) - will tune")
            logger.debug(f"  Cache path: {cache_file}")
            return None
        
        try:
            with open(cache_file) as f:
                cache_data = json.load(f)
            
            # Check age if specified
            if max_age_hours is not None:
                timestamp = datetime.fromisoformat(cache_data["timestamp"])
                age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                if age_hours > max_age_hours:
                    logger.warning(
                        f"Cached tuning for {optimizer} is {age_hours:.1f}h old "
                        f"(max={max_age_hours}h), ignoring"
                    )
                    return None
            
            logger.info(
                f"✅ Loaded cached tuning for {optimizer}: {cache_data['best_params']}"
            )
            return cache_data["best_params"]
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
            return None
    
    def clear_cache(self, dataset: Optional[str] = None):
        """
        Clear cached tuning results.
        
        Args:
            dataset: If specified, only clear cache for this dataset
        """
        if dataset:
            pattern = f"{dataset}_*.json"
        else:
            pattern = "*.json"
            
        removed = list(self.cache_dir.glob(pattern))
        for file in removed:
            file.unlink()
            
        logger.info(f"Cleared {len(removed)} cached tuning results")
        return removed


def create_tuning_cache(results_dir: Path) -> TuningCache:
    """
    Factory function to create TuningCache instance.
    
    Args:
        results_dir: Base results directory
        
    Returns:
        Configured TuningCache instance
    """
    cache_dir = Path(results_dir) / "tuning_cache"
    return TuningCache(cache_dir)
