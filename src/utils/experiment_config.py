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
    # Resume behavior controls how the runner behaves when --resume is requested but no checkpoint is found.
    # Valid choices:
    #  - 'error_if_no_checkpoint': raise an error when a checkpoint is missing
    #  - 'restart_if_no_checkpoint': proceed as a fresh run if no checkpoint exists
    #  - 'skip_if_results_exist': consult summary/results and skip if completed
    resume_behavior: str = None

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
        """Validate and normalize configuration after initialization."""
        # CRITICAL: Convert results_dir to Path and make ABSOLUTE
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        
        # ALWAYS resolve to absolute path to prevent CWD-dependent behavior
        if not self.results_dir.is_absolute():
            # Resolve relative to PROJECT ROOT, not current working directory
            # This file is in src/utils/, so project root is 2 levels up
            project_root = Path(__file__).parent.parent.parent
            self.results_dir = (project_root / self.results_dir).resolve()
        
        # Validate directory is writable (create if needed)
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise ValueError(
                f"results_dir {self.results_dir} is not writable: {e}\n"
                f"Check permissions or specify a different location with --output-dir"
            )

        # BUG FIX: Validate resume_behavior allowed values
        if self.resume_behavior is not None:
            allowed_behaviors = ['error_if_no_checkpoint', 'restart_if_no_checkpoint', 'skip_if_results_exist']
            if self.resume_behavior not in allowed_behaviors:
                raise ValueError(
                    f"Invalid resume_behavior '{self.resume_behavior}'. "
                    f"Must be one of: {allowed_behaviors}"
                )

        # Validate seeds (strict checks for statistical validity)
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
        """Create config from dictionary with strict validation.

        Backwards-compatibility: accept a single integer key 'seed' and
        migrate it to the canonical 'seeds' list.
        
        Enforces statistical validity by requiring minimum 3 seeds.
        """
        # EXPLICIT TYPE CONVERSION: str → Path BEFORE dataclass init
        if 'results_dir' in config_dict:
            results_dir = config_dict['results_dir']
            if isinstance(results_dir, str):
                config_dict['results_dir'] = Path(results_dir)
                # Path will be made absolute in __post_init__
        
        # Backwards compatibility: accept 'seed' as alias for 'seeds'
        if 'seed' in config_dict and 'seeds' not in config_dict:
            seed_val = config_dict.pop('seed')
            # If caller provided an integer, wrap it in a list
            if isinstance(seed_val, int):
                config_dict['seeds'] = [seed_val]
            # If caller already provided a list under 'seed', accept it
            elif isinstance(seed_val, (list, tuple)):
                config_dict['seeds'] = list(seed_val)

        # STRICT VALIDATION: Enforce minimum 3 seeds for statistical validity
        if 'seeds' in config_dict:
            seeds = config_dict['seeds']
            
            if not isinstance(seeds, (list, tuple)):
                raise TypeError(
                    f"'seeds' must be list or tuple, got {type(seeds).__name__}"
                )
            
            if len(seeds) < 3:
                raise ValueError(
                    f"STATISTICAL INTEGRITY ERROR: Got {len(seeds)} seeds: {seeds}\n\n"
                    f"MINIMUM 3 seeds required for:\n"
                    f"  - Variance estimation (σ²)\n"
                    f"  - Confidence intervals (t-test requires n ≥ 3)\n"
                    f"  - Reproducibility verification\n\n"
                    f"Recommended: 5+ seeds for robust statistics.\n"
                    f"See: https://en.wikipedia.org/wiki/Standard_deviation#Sample_standard_deviation\n\n"
                    f"All experiments must report mean ± std to comply with ML reproducibility standards."
                )
            
            if len(seeds) > 20:
                import logging
                logging.warning(
                    f"Large seed count ({len(seeds)}) detected. "
                    f"Consider reducing to 5-10 seeds unless conducting power analysis."
                )
            
            if len(seeds) != len(set(seeds)):
                duplicates = [s for s in seeds if seeds.count(s) > 1]
                raise ValueError(
                    f"DUPLICATE SEEDS: {duplicates}\n"
                    f"All seeds must be unique to ensure independent runs."
                )
            
            if any(not isinstance(s, int) or not (0 <= s < 2**32) for s in seeds):
                invalid = [s for s in seeds if not isinstance(s, int) or not (0 <= s < 2**32)]
                raise ValueError(
                    f"INVALID SEEDS: {invalid}\n"
                    f"Seeds must be integers in range [0, 2^32-1] for RNG compatibility."
                )

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

    # Resolve resume behavior default: if not explicitly provided use 'skip_if_results_exist' when resume is used,
    # otherwise default to 'restart_if_no_checkpoint'. This mirrors CLI behavior.
    if hasattr(args, 'resume_behavior') and getattr(args, 'resume_behavior') is not None:
        overrides['resume_behavior'] = args.resume_behavior
    else:
        overrides['resume_behavior'] = 'skip_if_results_exist' if getattr(args, 'resume', False) else 'restart_if_no_checkpoint'

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
