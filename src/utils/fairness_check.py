"""
Fairness Validation Utility for Hyperparameter Tuning.

This module ensures fair comparisons between optimizers by validating that
all compared algorithms receive equal computational budgets during hyperparameter
tuning. Detects and prevents tuning budget disparities that could bias results.

Key Features:
- Validates equal n_trials across all optimizers
- Checks for consistent evaluation protocols (epochs, batch sizes)
- Detects missing tuning for advanced optimizers
- Provides actionable error messages for fixing disparities

Usage:
    from src.utils.fairness_check import validate_tuning_fairness
    
    optimizers = ['SGD', 'Adam', 'SAM_SGD', 'Lookahead_Adam']
    tuning_configs = {
        'SGD': {'n_trials': 15, 'epochs': 3},
        'Adam': {'n_trials': 15, 'epochs': 3},
        'SAM_SGD': {'n_trials': 15, 'epochs': 3},
        'Lookahead_Adam': {'n_trials': 15, 'epochs': 3}
    }
    validate_tuning_fairness(optimizers, tuning_configs)  # Raises if unfair

References:
- Li et al. (2020): "Hyperparameter Optimization: A Spectral Approach"
- Bergstra & Bengio (2012): "Random Search for Hyper-Parameter Optimization"
"""

import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning of a single optimizer."""
    n_trials: int
    epochs: int
    batch_size: Optional[int] = None
    is_tuned: bool = True
    tuning_method: str = "optuna"  # 'optuna', 'grid', 'random', or 'default'
    
    @property
    def total_budget(self) -> int:
        """Compute total computational budget (trials × epochs)."""
        return self.n_trials * self.epochs


class TuningFairnessValidator:
    """
    Validates fairness of hyperparameter tuning across optimizers.
    
    Ensures that:
    1. All compared optimizers receive equal tuning budgets (n_trials)
    2. Evaluation protocols are consistent (epochs, batch sizes)
    3. Advanced optimizers are not disadvantaged by using defaults
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: If True, raises exceptions on violations. 
                        If False, only logs warnings.
        """
        self.strict_mode = strict_mode
        self.violations: List[str] = []
    
    def validate(
        self,
        optimizers: List[str],
        tuning_configs: Dict[str, TuningConfig]
    ) -> bool:
        """
        Validate tuning fairness across optimizers.
        
        Args:
            optimizers: List of optimizer names to compare
            tuning_configs: Dict mapping optimizer names to their tuning configs
            
        Returns:
            True if fair, False otherwise (or raises in strict mode)
            
        Raises:
            TuningFairnessError: If violations detected in strict mode
        """
        self.violations = []
        
        # Check 1: All optimizers have tuning configs
        self._check_missing_configs(optimizers, tuning_configs)
        
        # Check 2: Equal n_trials across all optimizers
        self._check_trial_parity(optimizers, tuning_configs)
        
        # Check 3: Consistent evaluation protocols
        self._check_protocol_consistency(optimizers, tuning_configs)
        
        # Check 4: No optimizers using defaults while others are tuned
        self._check_tuning_method_consistency(optimizers, tuning_configs)
        
        # Report results
        if self.violations:
            error_msg = self._format_violation_report()
            if self.strict_mode:
                raise TuningFairnessError(error_msg)
            else:
                logger.warning(f"Tuning fairness violations detected:\n{error_msg}")
                return False
        else:
            logger.info("✓ Tuning fairness validated: All optimizers have equal budgets")
            return True
    
    def _check_missing_configs(
        self,
        optimizers: List[str],
        tuning_configs: Dict[str, TuningConfig]
    ):
        """Check if any optimizers lack tuning configurations."""
        missing = [opt for opt in optimizers if opt not in tuning_configs]
        if missing:
            self.violations.append(
                f"Missing tuning configs for {len(missing)} optimizer(s): {missing}. "
                "All compared optimizers must have explicit tuning configurations."
            )
    
    def _check_trial_parity(
        self,
        optimizers: List[str],
        tuning_configs: Dict[str, TuningConfig]
    ):
        """Check if all optimizers have equal number of tuning trials."""
        trials_map = defaultdict(list)
        for opt in optimizers:
            if opt in tuning_configs:
                n_trials = tuning_configs[opt].n_trials
                trials_map[n_trials].append(opt)
        
        if len(trials_map) > 1:
            # Multiple different trial counts detected
            sorted_groups = sorted(trials_map.items(), key=lambda x: x[0], reverse=True)
            violation_details = []
            for n_trials, opts in sorted_groups:
                violation_details.append(f"  - {n_trials} trials: {opts}")
            
            self.violations.append(
                f"Unequal tuning budgets detected:\n" +
                "\n".join(violation_details) + "\n" +
                "All optimizers must receive equal n_trials for fair comparison."
            )
    
    def _check_protocol_consistency(
        self,
        optimizers: List[str],
        tuning_configs: Dict[str, TuningConfig]
    ):
        """Check if evaluation protocols (epochs, batch sizes) are consistent."""
        epochs_map = defaultdict(list)
        batch_size_map = defaultdict(list)
        
        for opt in optimizers:
            if opt in tuning_configs:
                config = tuning_configs[opt]
                epochs_map[config.epochs].append(opt)
                if config.batch_size is not None:
                    batch_size_map[config.batch_size].append(opt)
        
        # Check epoch consistency
        if len(epochs_map) > 1:
            details = [f"  - {ep} epochs: {opts}" for ep, opts in sorted(epochs_map.items())]
            self.violations.append(
                f"Inconsistent evaluation epochs:\n" + "\n".join(details) + "\n" +
                "All trials should use the same number of epochs."
            )
        
        # Check batch size consistency (if specified)
        if len(batch_size_map) > 1:
            details = [f"  - batch_size={bs}: {opts}" for bs, opts in sorted(batch_size_map.items())]
            self.violations.append(
                f"Inconsistent batch sizes:\n" + "\n".join(details) + "\n" +
                "Different batch sizes affect gradient noise and may bias comparisons."
            )
    
    def _check_tuning_method_consistency(
        self,
        optimizers: List[str],
        tuning_configs: Dict[str, TuningConfig]
    ):
        """Check if some optimizers use defaults while others are tuned."""
        tuned_opts = [opt for opt in optimizers 
                     if opt in tuning_configs and tuning_configs[opt].is_tuned]
        default_opts = [opt for opt in optimizers 
                       if opt in tuning_configs and not tuning_configs[opt].is_tuned]
        
        if tuned_opts and default_opts:
            self.violations.append(
                f"Mixed tuning approaches:\n"
                f"  - Tuned optimizers ({len(tuned_opts)}): {tuned_opts}\n"
                f"  - Default hyperparameters ({len(default_opts)}): {default_opts}\n"
                "All compared optimizers must either be tuned or use defaults consistently."
            )
    
    def _format_violation_report(self) -> str:
        """Format violations into a readable report."""
        header = "=" * 70
        report = [
            header,
            "TUNING FAIRNESS VIOLATIONS DETECTED",
            header,
            "",
            "The following issues compromise fair optimizer comparison:",
            ""
        ]
        
        for i, violation in enumerate(self.violations, 1):
            report.append(f"{i}. {violation}")
            report.append("")
        
        report.extend([
            "RECOMMENDED FIXES:",
            "1. Ensure all optimizers have equal n_trials in tuning configs",
            "2. Use consistent epochs and batch sizes across all trials",
            "3. Either tune all optimizers or use defaults for all (no mixing)",
            "4. Document any exceptions with justification",
            "",
            header
        ])
        
        return "\n".join(report)


class TuningFairnessError(Exception):
    """Raised when tuning fairness violations are detected in strict mode."""
    pass


def validate_tuning_fairness(
    optimizers: List[str],
    tuning_configs: Dict[str, Dict[str, Any]],
    strict: bool = True
) -> bool:
    """
    Convenience function to validate tuning fairness.
    
    Args:
        optimizers: List of optimizer names to compare
        tuning_configs: Dict with tuning parameters for each optimizer
            e.g., {'SGD': {'n_trials': 15, 'epochs': 3}, ...}
        strict: If True, raises on violations. If False, only warns.
        
    Returns:
        True if fair, False otherwise
        
    Raises:
        TuningFairnessError: If violations detected in strict mode
        
    Example:
        >>> optimizers = ['SGD', 'Adam', 'SAM_SGD']
        >>> configs = {
        ...     'SGD': {'n_trials': 15, 'epochs': 3, 'is_tuned': True},
        ...     'Adam': {'n_trials': 15, 'epochs': 3, 'is_tuned': True},
        ...     'SAM_SGD': {'n_trials': 5, 'epochs': 3, 'is_tuned': True}  # UNFAIR!
        ... }
        >>> validate_tuning_fairness(optimizers, configs)  # Raises error
    """
    # Convert dict configs to TuningConfig objects
    config_objects = {}
    for opt_name, config_dict in tuning_configs.items():
        config_objects[opt_name] = TuningConfig(
            n_trials=config_dict.get('n_trials', 0),
            epochs=config_dict.get('epochs', 0),
            batch_size=config_dict.get('batch_size'),
            is_tuned=config_dict.get('is_tuned', True),
            tuning_method=config_dict.get('tuning_method', 'optuna')
        )
    
    validator = TuningFairnessValidator(strict_mode=strict)
    return validator.validate(optimizers, config_objects)


def check_tuning_parity_in_results(
    results_df,
    optimizer_col: str = 'optimizer',
    trial_col: Optional[str] = None
) -> bool:
    """
    Post-hoc check for tuning parity from experiment results DataFrame.
    
    Args:
        results_df: DataFrame with experiment results
        optimizer_col: Column name containing optimizer names
        trial_col: Column name with trial numbers (if available)
        
    Returns:
        True if parity detected, False otherwise
    """
    if trial_col and trial_col in results_df.columns:
        # Count unique trials per optimizer
        trial_counts = results_df.groupby(optimizer_col)[trial_col].nunique()
        
        if trial_counts.nunique() > 1:
            logger.warning(
                f"Tuning parity violation detected in results:\n{trial_counts}\n"
                "Different optimizers have different numbers of trials."
            )
            return False
    
    logger.info("No tuning parity violations detected in results DataFrame")
    return True


def generate_fair_tuning_config(
    optimizers: List[str],
    n_trials: int = 15,
    epochs: int = 3,
    batch_size: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Generate a fair tuning configuration for multiple optimizers.
    
    Args:
        optimizers: List of optimizer names
        n_trials: Number of tuning trials (same for all)
        epochs: Number of epochs per trial (same for all)
        batch_size: Batch size (if relevant)
        
    Returns:
        Dict with identical configs for all optimizers
        
    Example:
        >>> optimizers = ['SGD', 'Adam', 'SAM_SGD', 'Lookahead_Adam']
        >>> config = generate_fair_tuning_config(optimizers, n_trials=20, epochs=5)
        >>> validate_tuning_fairness(optimizers, config)  # Passes
    """
    config = {}
    for opt in optimizers:
        config[opt] = {
            'n_trials': n_trials,
            'epochs': epochs,
            'batch_size': batch_size,
            'is_tuned': True,
            'tuning_method': 'optuna'
        }
    return config


if __name__ == "__main__":
    # Example: Detecting unfair tuning
    optimizers = ['SGD', 'Adam', 'AdamW', 'SAM_SGD', 'Lookahead_Adam']
    
    # UNFAIR configuration (SAM_SGD and Lookahead_Adam disadvantaged)
    unfair_config = {
        'SGD': {'n_trials': 15, 'epochs': 3, 'is_tuned': True},
        'Adam': {'n_trials': 15, 'epochs': 3, 'is_tuned': True},
        'AdamW': {'n_trials': 15, 'epochs': 3, 'is_tuned': True},
        'SAM_SGD': {'n_trials': 0, 'epochs': 0, 'is_tuned': False},  # Using defaults!
        'Lookahead_Adam': {'n_trials': 0, 'epochs': 0, 'is_tuned': False}  # Using defaults!
    }
    
    print("Testing UNFAIR configuration:")
    try:
        validate_tuning_fairness(optimizers, unfair_config, strict=True)
    except TuningFairnessError as e:
        print(f"✓ Correctly detected unfairness:\n{e}\n")
    
    # FAIR configuration
    fair_config = generate_fair_tuning_config(optimizers, n_trials=15, epochs=3)
    
    print("\nTesting FAIR configuration:")
    result = validate_tuning_fairness(optimizers, fair_config, strict=True)
    print(f"✓ Validation passed: {result}")
