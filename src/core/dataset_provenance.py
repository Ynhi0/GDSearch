"""
Dataset Provenance Tracking for Research Reproducibility

This module provides standardized logging of dataset metadata to MLflow
to ensure full reproducibility and traceability of experiments.

Addresses research proposal requirement: "Ensure every experiment logs
dataset name, version, seed, and config in MLflow for scientific rigor."
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    import torchvision
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import medmnist
    HAS_MEDMNIST = True
except ImportError:
    HAS_MEDMNIST = False

try:
    import datasets  # HuggingFace datasets
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False


def get_dataset_provenance(
    dataset_name: str,
    split: str = 'train',
    data_root: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract dataset provenance metadata for logging.
    
    Args:
        dataset_name: Name of dataset (MNIST, CIFAR-10, MedMNIST, etc.)
        split: Dataset split (train/val/test)
        data_root: Root directory where data is stored
        **kwargs: Additional dataset-specific parameters
    
    Returns:
        Dict with provenance metadata:
            - dataset_name: Official name
            - dataset_version: Version/checksum if available
            - split: Train/val/test
            - num_samples: Number of samples in split
            - data_source: Source library (torchvision, medmnist, etc.)
            - timestamp: ISO timestamp of provenance extraction
            - data_root: Path to data directory
            - checksums: Dict of file checksums if available
    """
    provenance = {
        'dataset_name': dataset_name,
        'split': split,
        'data_source': 'unknown',
        'timestamp': datetime.utcnow().isoformat(),
        'data_root': str(data_root) if data_root else None,
    }
    
    dataset_upper = dataset_name.upper()
    
    # MNIST
    if dataset_upper == 'MNIST':
        provenance['data_source'] = 'torchvision'
        provenance['dataset_version'] = f'torchvision_{torchvision.__version__}' if HAS_TORCH else 'unknown'
        provenance['official_url'] = 'http://yann.lecun.com/exdb/mnist/'
        
        # Try to get sample count
        if HAS_TORCH and data_root:
            try:
                ds = torchvision.datasets.MNIST(root=data_root, train=(split=='train'), download=False)
                provenance['num_samples'] = len(ds)
            except Exception as e:
                logger.debug(f"Could not load MNIST for sample count: {e}")
                provenance['num_samples'] = 60000 if split == 'train' else 10000  # Known sizes
    
    # CIFAR-10
    elif dataset_upper in ['CIFAR-10', 'CIFAR10']:
        provenance['data_source'] = 'torchvision'
        provenance['dataset_version'] = f'torchvision_{torchvision.__version__}' if HAS_TORCH else 'unknown'
        provenance['official_url'] = 'https://www.cs.toronto.edu/~kriz/cifar.html'
        
        if HAS_TORCH and data_root:
            try:
                ds = torchvision.datasets.CIFAR10(root=data_root, train=(split=='train'), download=False)
                provenance['num_samples'] = len(ds)
            except Exception:
                provenance['num_samples'] = 50000 if split == 'train' else 10000
    
    # CIFAR-100
    elif dataset_upper in ['CIFAR-100', 'CIFAR100']:
        provenance['data_source'] = 'torchvision'
        provenance['dataset_version'] = f'torchvision_{torchvision.__version__}' if HAS_TORCH else 'unknown'
        provenance['official_url'] = 'https://www.cs.toronto.edu/~kriz/cifar.html'
        
        if HAS_TORCH and data_root:
            try:
                ds = torchvision.datasets.CIFAR100(root=data_root, train=(split=='train'), download=False)
                provenance['num_samples'] = len(ds)
            except Exception:
                provenance['num_samples'] = 50000 if split == 'train' else 10000
    
    # FashionMNIST
    elif dataset_upper in ['FASHIONMNIST', 'FASHION-MNIST']:
        provenance['data_source'] = 'torchvision'
        provenance['dataset_version'] = f'torchvision_{torchvision.__version__}' if HAS_TORCH else 'unknown'
        provenance['official_url'] = 'https://github.com/zalandoresearch/fashion-mnist'
        
        if HAS_TORCH and data_root:
            try:
                ds = torchvision.datasets.FashionMNIST(root=data_root, train=(split=='train'), download=False)
                provenance['num_samples'] = len(ds)
            except Exception:
                provenance['num_samples'] = 60000 if split == 'train' else 10000
    
    # MedMNIST family
    elif 'MEDMNIST' in dataset_upper or dataset_upper in ['PATHMNIST', 'CHESTMNIST', 'DERMAMNIST', 'OCTMNIST', 'PNEUMONIAMNIST', 'RETINAMNIST', 'BREASTMNIST', 'BLOODMNIST', 'TISSUEMNIST', 'ORGANAMNIST', 'ORGANCMNIST', 'ORGANSMNIST']:
        provenance['data_source'] = 'medmnist'
        provenance['dataset_version'] = f'medmnist_{medmnist.__version__}' if HAS_MEDMNIST else 'unknown'
        provenance['official_url'] = 'https://medmnist.com/'
        provenance['citation'] = 'Yang et al. MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. arXiv:2110.14795'
        
        # MedMNIST datasets have known splits
        # Default sizes (can vary by specific dataset)
        provenance['medmnist_note'] = 'Sizes vary by specific MedMNIST subset'
    
    # IMDB (NLP)
    elif dataset_upper == 'IMDB':
        provenance['data_source'] = 'huggingface_datasets'
        provenance['dataset_version'] = f'datasets_{datasets.__version__}' if HAS_HF_DATASETS else 'unknown'
        provenance['official_url'] = 'https://ai.stanford.edu/~amaas/data/sentiment/'
        provenance['citation'] = 'Maas et al. Learning Word Vectors for Sentiment Analysis. ACL 2011'
        
        if HAS_HF_DATASETS:
            try:
                from datasets import load_dataset
                ds = load_dataset('imdb', split=split)
                provenance['num_samples'] = len(ds)
            except Exception:
                provenance['num_samples'] = {'train': 25000, 'test': 25000}.get(split, 'unknown')
    
    # Synthetic datasets
    elif 'SYNTHETIC' in dataset_upper:
        provenance['data_source'] = 'synthetic'
        provenance['dataset_version'] = 'generated'
        provenance['warning'] = 'SYNTHETIC DATA - Not suitable for publication claims'
        provenance['num_samples'] = kwargs.get('num_samples', 'unknown')
    
    # Unknown dataset
    else:
        provenance['warning'] = f'Unknown dataset type: {dataset_name}'
        logger.warning(f"Dataset provenance unknown for: {dataset_name}")
    
    # Add checksum if data_root available
    if data_root and Path(data_root).exists():
        try:
            provenance['data_root_checksum'] = _compute_directory_checksum(Path(data_root) / dataset_name)
        except Exception as e:
            logger.debug(f"Could not compute checksum: {e}")
    
    return provenance


def log_dataset_provenance_to_mlflow(
    mlflow_logger: Any,
    dataset_name: str,
    split: str = 'train',
    data_root: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs
) -> None:
    """
    Log dataset provenance to MLflow experiment tracker.
    
    Args:
        mlflow_logger: MLflow logger instance with log_params method
        dataset_name: Name of dataset
        split: Dataset split
        data_root: Data root directory
        seed: Random seed used
        **kwargs: Additional metadata to log
    """
    provenance = get_dataset_provenance(dataset_name, split, data_root, **kwargs)
    
    # Add seed if provided
    if seed is not None:
        provenance['data_seed'] = seed
    
    # Flatten provenance for MLflow params (mlflow doesn't handle nested dicts)
    flattened = {}
    for key, value in provenance.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened[f'dataset_{key}_{subkey}'] = str(subvalue)
        else:
            flattened[f'dataset_{key}'] = str(value)
    
    # Log to MLflow
    if hasattr(mlflow_logger, 'log_params'):
        mlflow_logger.log_params(flattened)
    else:
        logger.warning("MLflow logger does not have log_params method")


def _compute_directory_checksum(directory: Path, max_files: int = 10) -> str:
    """
    Compute a simple checksum of a directory's contents.
    
    Args:
        directory: Path to directory
        max_files: Maximum number of files to hash (to avoid long computation)
    
    Returns:
        Hex digest of combined file checksums
    """
    if not directory.exists():
        return 'directory_not_found'
    
    hasher = hashlib.md5()
    files = sorted(directory.glob('*'))[:max_files]
    
    for file_path in files:
        if file_path.is_file() and file_path.stat().st_size < 10 * 1024 * 1024:  # Skip files > 10MB
            try:
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
            except Exception:
                pass  # Skip unreadable files
    
    return hasher.hexdigest()[:16]  # First 16 chars


def create_experiment_manifest(
    experiment_name: str,
    config: Dict[str, Any],
    dataset_provenance: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Create a JSON manifest file with full experiment metadata.
    
    This provides a self-contained record of the experiment setup
    for reproducibility and auditing.
    
    Args:
        experiment_name: Name of experiment
        config: Full experiment configuration dict
        dataset_provenance: Dataset provenance from get_dataset_provenance()
        output_path: Path to save manifest JSON
    """
    manifest = {
        'experiment_name': experiment_name,
        'created_at': datetime.utcnow().isoformat(),
        'config': config,
        'dataset_provenance': dataset_provenance,
        'environment': {
            'torch_version': torch.__version__ if HAS_TORCH else None,
            'torchvision_version': torchvision.__version__ if HAS_TORCH else None,
            'medmnist_version': medmnist.__version__ if HAS_MEDMNIST else None,
            'datasets_version': datasets.__version__ if HAS_HF_DATASETS else None,
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Experiment manifest saved to {output_path}")


if __name__ == '__main__':
    # Demo
    print("=== Dataset Provenance Tracker Demo ===\n")
    
    # Test MNIST provenance
    mnist_prov = get_dataset_provenance('MNIST', split='train', data_root='./data')
    print("MNIST Provenance:")
    for k, v in mnist_prov.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*50 + "\n")
    
    # Test MedMNIST provenance
    med_prov = get_dataset_provenance('PathMNIST', split='train')
    print("PathMNIST Provenance:")
    for k, v in med_prov.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*50 + "\n")
    
    # Test synthetic warning
    synth_prov = get_dataset_provenance('SyntheticMedical', split='train', num_samples=5000)
    print("Synthetic Dataset Provenance:")
    for k, v in synth_prov.items():
        print(f"  {k}: {v}")
    
    if 'warning' in synth_prov:
        print(f"\n⚠️  WARNING: {synth_prov['warning']}")
