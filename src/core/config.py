"""
Configuration dataclasses for GDSearch experiments.

Replaces global variables with structured configuration objects.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution.
    
    This replaces all global variables from run_all_kaggle.py with a structured,
    type-safe configuration object that can be passed explicitly to functions.
    """
    
    # Auto-tuning features (replaces AUTO_LR_ENABLED, ADAPTIVE_BATCH_ENABLED, etc.)
    auto_lr_enabled: bool = False
    adaptive_batch_enabled: bool = False
    ultra_quick_mode: bool = False
    
    # Training enhancements (replaces USE_AMP, USE_EMA, LABEL_SMOOTHING)
    use_amp: bool = False
    use_ema: bool = False
    label_smoothing: float = 0.0
    
    # Experiment settings
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    quick: bool = False
    skip_tuning: bool = False
    resume: bool = False
    deterministic: bool = False
    
    # Results directory
    results_dir: Path = field(default_factory=lambda: Path("results"))
    
    # Kaggle optimizations
    kaggle_t4: bool = False
    kaggle_config: Optional['KaggleConfig'] = None
    
    # Profiling and tracking (replaces profiler, tracker, checkpoint_manager globals)
    profile: bool = False
    no_mlflow: bool = False
    
    # Time budget (hours)
    max_hours: float = 11.0
    warning_hours: float = 10.5
    
    # Failed experiments tracking (replaces FAILED_EXPERIMENTS global list)
    failed_experiments: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        
        if self.ultra_quick_mode:
            # Auto-adjust settings for ultra-quick mode
            self.quick = True
            self.skip_tuning = True
    
    def get_epochs(self, default: int, quick_value: int) -> int:
        """Get number of epochs based on mode.
        
        Args:
            default: Default epoch count for full run
            quick_value: Epoch count for quick mode
            
        Returns:
            Number of epochs to use
        """
        if self.ultra_quick_mode:
            return 2
        elif self.quick:
            return quick_value
        else:
            return default
    
    def add_failed_experiment(self, experiment_name: str, error: Exception, context: str = ""):
        """Track a failed experiment.
        
        Args:
            experiment_name: Name of the failed experiment
            error: Exception that caused the failure
            context: Additional context about the failure
        """
        self.failed_experiments.append({
            'experiment': experiment_name,
            'error': str(error),
            'context': context,
            'type': type(error).__name__
        })


@dataclass
class KaggleConfig:
    """Configuration for Kaggle T4 optimizations."""
    
    cudnn_benchmark: bool = True
    use_amp: bool = True
    pin_memory: bool = True
    num_workers: int = 4
    persistent_workers: bool = True
    
    # Batch sizes per experiment type
    batch_size_mnist: int = 256
    batch_size_cifar10: int = 128
    batch_size_resnet: int = 128
    batch_size_nlp: int = 32
    batch_size_medical: int = 8
    
    def get_batch_size(self, experiment_type: str, default: int = 128) -> int:
        """Get batch size for experiment type.
        
        Args:
            experiment_type: Type of experiment (mnist, cifar10, etc.)
            default: Default batch size if not configured
            
        Returns:
            Configured or default batch size
        """
        key = f'batch_size_{experiment_type}'
        return getattr(self, key, default)
