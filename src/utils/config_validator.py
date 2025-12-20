"""
Config validation utilities for GDSearch experiments.

Ensures config files use consistent keys and valid values.
Detects "zombie keys" (present but unused) and enforces schema compliance.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
import warnings
import logging


class TrackedConfig(dict):
    """
    Config dict that tracks which keys are accessed.
    
    Use this to automatically detect zombie keys during experiment execution.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._accessed = set()
    
    def __getitem__(self, key):
        self._accessed.add(key)
        return super().__getitem__(key)
    
    def get(self, key, default=None):
        if key in self:
            self._accessed.add(key)
        return super().get(key, default)
    
    def get_zombies(self) -> Set[str]:
        """Return keys that exist but were never accessed."""
        return set(self.keys()) - self._accessed
    
    def report_zombies(self) -> None:
        """Log warning about zombie config keys."""
        zombies = self.get_zombies()
        if zombies:
            logging.warning(
                f"ZOMBIE CONFIG KEYS (present but unused): {zombies}. "
                f"These values may not affect experiments!"
            )


def validate_config_keys(config_path: str, strict: bool = False) -> Dict[str, List[str]]:
    """
    Validate config file uses standardized keys.
    
    Args:
        config_path: Path to config JSON file
        strict: If True, treat warnings as errors
        
    Returns:
        Dictionary with 'warnings' and 'errors' lists
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    issues = {'warnings': [], 'errors': [], 'zombies': []}
    
    # Define expected keys for different config types
    expected_sweep_keys = {
        'optimizers', 'lr_values', 'momentum_values', 'beta1_values',
        'beta2_values', 'weight_decay_values', 'epsilon_values', 'name'
    }
    
    expected_top_keys = {
        'dataset', 'model_type', 'sweeps', 'epochs', 'batch_size',
        'seeds', 'device', 'patience', 'save_checkpoints', 'data_augmentation',
        'optimizer', 'lr_values'  # Old format compatibility
    }
    
    # Check for zombie keys at top level
    actual_top_keys = set(config.keys())
    potential_zombies = actual_top_keys - expected_top_keys
    if potential_zombies:
        issues['zombies'].extend([f"Top-level: {k}" for k in potential_zombies])
        issues['warnings'].append(
            f"Unexpected top-level keys (may be ignored): {potential_zombies}"
        )
    
    # Check for deprecated/inconsistent keys
    for sweep_idx, sweep in enumerate(config.get('sweeps', [])):
        # Check for zombie keys in sweep
        actual_sweep_keys = set(sweep.keys())
        sweep_zombies = actual_sweep_keys - expected_sweep_keys
        if sweep_zombies:
            issues['zombies'].extend([f"Sweep[{sweep_idx}]: {k}" for k in sweep_zombies])
        
        for opt_config in sweep.get('optimizers', []):
            # Handle both list and dict format for optimizers
            if isinstance(opt_config, dict):
                # Check for learning_rates vs lr_values
                has_learning_rates = 'learning_rates' in opt_config
                has_lr_values = 'lr_values' in opt_config
                
                if has_learning_rates and has_lr_values:
                    issues['warnings'].append(
                        f"Optimizer {opt_config.get('name')} has both 'learning_rates' and 'lr_values'. "
                        f"Using 'lr_values' (preferred)."
                    )
                elif has_learning_rates and not has_lr_values:
                    issues['warnings'].append(
                        f"Optimizer {opt_config.get('name')} uses deprecated key 'learning_rates'. "
                        f"Consider renaming to 'lr_values'."
                    )
                elif not has_learning_rates and not has_lr_values:
                    # Check if lr_values is at sweep level
                    if 'lr_values' not in sweep:
                        issues['errors'].append(
                            f"Optimizer {opt_config.get('name')} missing required key 'lr_values' "
                            f"(not in optimizer config or sweep level)."
                        )
    
    # In strict mode, zombies become errors
    if strict and issues['zombies']:
        issues['errors'].append(
            f"STRICT MODE: Zombie config keys detected: {issues['zombies']}"
        )
    
    return issues


def normalize_config(config_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Normalize config file to use standard keys.
    
    Args:
        config_path: Path to input config JSON
        output_path: Path to save normalized config (defaults to overwriting input)
        
    Returns:
        Normalized config dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Normalize keys
    for sweep in config.get('sweeps', []):
        for opt_config in sweep.get('optimizers', []):
            # Rename learning_rates → lr_values
            if 'learning_rates' in opt_config and 'lr_values' not in opt_config:
                opt_config['lr_values'] = opt_config.pop('learning_rates')
    
    # Save normalized config
    if output_path is None:
        output_path = config_path
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    return config


def validate_all_configs(config_dir: str = 'configs', strict: bool = False) -> bool:
    """
    Validate all config files in directory.
    
    Args:
        config_dir: Directory containing config JSON files
        strict: If True, treat warnings as errors
        
    Returns:
        True if all configs valid, False otherwise
    """
    config_path = Path(config_dir)
    all_valid = True
    
    for config_file in config_path.glob('*.json'):
        if config_file.name == 'config_schema.json':
            continue
        
        print(f"\nValidating {config_file.name}...")
        issues = validate_config_keys(str(config_file), strict=strict)
        
        if issues['errors']:
            print(f"  ERRORS:")
            for err in issues['errors']:
                print(f"     - {err}")
            all_valid = False
        
        if issues['zombies']:
            print(f"  ZOMBIE KEYS (may be ignored by code):")
            for zombie in issues['zombies']:
                print(f"     - {zombie}")
            if strict:
                all_valid = False
        
        if issues['warnings']:
            print(f"  WARNINGS:")
            for warn in issues['warnings']:
                print(f"     - {warn}")
            if strict:
                all_valid = False
        
        if not issues['errors'] and not issues['warnings'] and not issues['zombies']:
            print(f"  Valid")
    
    return all_valid


def load_tracked_config(config_path: str) -> 'TrackedConfig':
    """
    Load config as TrackedConfig to detect zombie keys during execution.
    
    Args:
        config_path: Path to config JSON
        
    Returns:
        TrackedConfig instance that monitors key access
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return TrackedConfig(config)


if __name__ == '__main__':
    import sys
    
    # Check for --strict flag
    strict = '--strict' in sys.argv
    if strict:
        sys.argv.remove('--strict')
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        issues = validate_config_keys(config_path, strict=strict)
        
        if issues['zombies']:
            print("ZOMBIE KEYS:")
            for zombie in issues['zombies']:
                print(f"  - {zombie}")
        
        if issues['errors']:
            print("ERRORS:")
            for err in issues['errors']:
                print(f"  - {err}")
            sys.exit(1)
        
        if issues['warnings']:
            print("WARNINGS:")
            for warn in issues['warnings']:
                print(f"  - {warn}")
            if strict:
                sys.exit(1)
    else:
        # Validate all configs
        print("Validating all configs in configs/...")
        if strict:
            print("(STRICT MODE: warnings and zombies treated as errors)")
        if not validate_all_configs(strict=strict):
            sys.exit(1)
        print("\nAll configs validated successfully!")
