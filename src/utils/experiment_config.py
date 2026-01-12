"""
Experiment Configuration Management

Provides a centralized configuration system using dataclasses to replace
scattered global flags and improve type safety and documentation.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class ExperimentConfig:
    """
    Centralized configuration for GDSearch experiments.

    Replaces scattered global flags (ULTRA_QUICK_MODE, AUTO_LR_ENABLED, etc.)
    with a typed, documented, and easily serializable configuration object.
    """

    # Execution mode
    ultra_quick_mode: bool = False
    quick_mode: bool = False
    resume: bool = False

    # Seeds for reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011])

    # Tuning configuration
    skip_tuning: bool = False
    n_trials: int = 15
    tune_epochs: int = 3

    # Training enhancements
    auto_lr_enabled: bool = False
    adaptive_batch_enabled: bool = False
    use_checkpointing: bool = True
    use_ema: bool = False
    use_mixed_precision: bool = False

    # Experiment selection
    selected_experiments: List[str] = field(default_factory=lambda: [
        'mnist', 'cifar10', 'medical', 'nlp',
        'scheduler_ablation', 'label_noise'
    ])

    # Resource limits
    max_epochs: int = 50
    time_budget_hours: Optional[float] = None
    checkpoint_interval: int = 5

    # Output configuration
    results_dir: Path = field(default_factory=lambda: Path('results'))
    save_plots: bool = True
    save_checkpoints: bool = True

    # Reproducibility
    exclude_tainted: bool = True  # Exclude OOM-tainted runs from aggregation
    validate_fairness: bool = True  # Check tuning budget parity
    strict_fairness: bool = False  # Raise on fairness violations

    # Logging
    log_level: str = 'INFO'
    log_to_file: bool = True
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert results_dir to Path if string
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)

        # Validate seeds
        if not self.seeds:
            raise ValueError("At least one seed must be specified")

        # Validate mode consistency
        if self.ultra_quick_mode:
            self.max_epochs = min(self.max_epochs, 2)
            self.n_trials = min(self.n_trials, 5)

        if self.quick_mode:
            self.max_epochs = min(self.max_epochs, 20)
            self.n_trials = min(self.n_trials, 10)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Filter out unknown keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    @classmethod
    def from_json(cls, filepath: Path) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        # Convert Path to string for JSON serialization
        config_dict['results_dir'] = str(config_dict['results_dir'])
        return config_dict

    def to_json(self, filepath: Path):
        """Save config to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_effective_epochs(self) -> int:
        """Get effective number of epochs based on mode."""
        if self.ultra_quick_mode:
            return 2
        elif self.quick_mode:
            return 20
        else:
            return self.max_epochs

    def get_effective_n_trials(self) -> int:
        """Get effective number of tuning trials based on mode."""
        if self.skip_tuning:
            return 0
        elif self.ultra_quick_mode:
            return 5
        elif self.quick_mode:
            return 10
        else:
            return self.n_trials

    def summary(self) -> str:
        """Get human-readable configuration summary."""
        lines = [
            "Experiment Configuration",
            "=" * 80,
            f"Mode: {'ULTRA QUICK' if self.ultra_quick_mode else 'QUICK' if self.quick_mode else 'FULL'}",
            f"Seeds: {self.seeds}",
            f"Epochs: {self.get_effective_epochs()}",
            f"Tuning: {'DISABLED' if self.skip_tuning else f'{self.get_effective_n_trials()} trials'}",
            f"Selected experiments: {', '.join(self.selected_experiments)}",
            "",
            "Features:",
            f"  Auto LR: {self.auto_lr_enabled}",
            f"  Adaptive batch: {self.adaptive_batch_enabled}",
            f"  Checkpointing: {self.use_checkpointing}",
            f"  Mixed precision: {self.use_mixed_precision}",
            "",
            "Reproducibility:",
            f"  Exclude tainted runs: {self.exclude_tainted}",
            f"  Validate fairness: {self.validate_fairness}",
            f"  Strict fairness: {self.strict_fairness}",
            "",
            f"Results directory: {self.results_dir}",
            "=" * 80
        ]
        return "\n".join(lines)


# Predefined configurations for common use cases
ULTRA_QUICK_CONFIG = ExperimentConfig(
    ultra_quick_mode=True,
    quick_mode=False,
    seeds=[42],
    max_epochs=2,
    n_trials=5,
    selected_experiments=['mnist']
)

QUICK_CONFIG = ExperimentConfig(
    ultra_quick_mode=False,
    quick_mode=True,
    seeds=[42, 123, 456],
    max_epochs=20,
    n_trials=10,
    selected_experiments=['mnist', 'cifar10']
)

FULL_CONFIG = ExperimentConfig(
    ultra_quick_mode=False,
    quick_mode=False,
    seeds=[42, 123, 456, 789, 1011],
    max_epochs=50,
    n_trials=15,
    selected_experiments=['mnist', 'cifar10', 'medical', 'nlp']
)

DEVELOPMENT_CONFIG = ExperimentConfig(
    ultra_quick_mode=True,
    seeds=[42],
    skip_tuning=True,
    max_epochs=2,
    selected_experiments=['mnist'],
    save_checkpoints=False,
    validate_fairness=False
)


def get_config_from_args(args) -> ExperimentConfig:
    """
    Create ExperimentConfig from argparse arguments.

    Args:
        args: argparse.Namespace object

    Returns:
        ExperimentConfig instance
    """
    # Start with appropriate base config
    if hasattr(args, 'ultra_quick') and args.ultra_quick:
        config = ULTRA_QUICK_CONFIG
    elif hasattr(args, 'quick') and args.quick:
        config = QUICK_CONFIG
    else:
        config = FULL_CONFIG

    # Override with command-line arguments
    overrides = {}

    if hasattr(args, 'resume'):
        overrides['resume'] = args.resume

    if hasattr(args, 'seeds'):
        overrides['seeds'] = args.seeds

    if hasattr(args, 'experiments'):
        overrides['selected_experiments'] = args.experiments

    if hasattr(args, 'results_dir'):
        overrides['results_dir'] = Path(args.results_dir)

    if hasattr(args, 'skip_tuning'):
        overrides['skip_tuning'] = args.skip_tuning

    # Create new config with overrides
    config_dict = config.to_dict()
    config_dict.update(overrides)

    return ExperimentConfig.from_dict(config_dict)


if __name__ == '__main__':
    # Demo: Show different configurations
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION DEMO")
    print("="*80 + "\n")

    print("1. ULTRA QUICK MODE")
    print("-" * 80)
    print(ULTRA_QUICK_CONFIG.summary())

    print("\n2. QUICK MODE")
    print("-" * 80)
    print(QUICK_CONFIG.summary())

    print("\n3. FULL MODE")
    print("-" * 80)
    print(FULL_CONFIG.summary())

    print("\n4. DEVELOPMENT MODE")
    print("-" * 80)
    print(DEVELOPMENT_CONFIG.summary())

    # Demo: Save and load
    print("\n" + "="*80)
    print("SAVE/LOAD DEMO")
    print("="*80)

    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.json"

        # Save
        FULL_CONFIG.to_json(config_path)
        print(f"✓ Saved config to {config_path}")

        # Load
        loaded_config = ExperimentConfig.from_json(config_path)
        print(f"✓ Loaded config: {loaded_config.get_effective_epochs()} epochs, {len(loaded_config.seeds)} seeds")

    print("\n" + "="*80)
    print("CONFIGURATION SYSTEM READY")
    print("="*80)
