#!/usr/bin/env python
"""
Configuration Validation Script - Zombie Key Detection

Detects unused JSON configuration keys that are silently ignored by code,
which can lead to debugging nightmares when typos cause silent failures.

Usage:
    python scripts/validate_configs.py
    python scripts/validate_configs.py --config configs/nn_tuning.json

Author: GDSearch Audit Remediation Team
Date: December 9, 2025
"""

import json
import argparse
from pathlib import Path
import re


class ConfigValidator:
    """Validate configuration files against actual code usage."""
    
    def __init__(self, project_root=None):
        if project_root is None:
            project_root = Path(__file__).parent.parent
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / 'src'
        self.scripts_dir = self.project_root / 'scripts'
    
    def find_zombie_keys(self, config_path, usage_dirs=None):
        """
        Find JSON keys that are never accessed in the codebase.
        
        Args:
            config_path: Path to JSON config file
            usage_dirs: Directories to search for key usage (default: src/ and scripts/)
        
        Returns:
            dict: {
                'zombie_keys': [list of unused keys],
                'used_keys': [list of keys found in code],
                'config_keys': [all keys in config]
            }
        """
        if usage_dirs is None:
            usage_dirs = [self.src_dir, self.scripts_dir]
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract all keys (including nested)
        config_keys = self._extract_all_keys(config)
        
        # Search for key usage in code
        used_keys = set()
        for directory in usage_dirs:
            if not directory.exists():
                continue
            
            for py_file in directory.rglob('*.py'):
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                for key in config_keys:
                    # Search for key as string literal or dict access
                    patterns = [
                        f'"{key}"',
                        f"'{key}'",
                        f'["{key}"]',
                        f"['{key}']",
                        f'.get("{key}"',
                        f".get('{key}'",
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            used_keys.add(key)
                            break
        
        zombie_keys = [k for k in config_keys if k not in used_keys]
        
        return {
            'zombie_keys': zombie_keys,
            'used_keys': sorted(used_keys),
            'config_keys': sorted(config_keys),
            'zombie_count': len(zombie_keys),
            'usage_rate': len(used_keys) / max(len(config_keys), 1) * 100
        }
    
    def _extract_all_keys(self, obj, prefix=''):
        """Recursively extract all keys from nested JSON."""
        keys = set()
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                keys.add(key)
                # Also check nested keys
                if isinstance(value, dict):
                    nested = self._extract_all_keys(value, prefix=f'{prefix}{key}.')
                    keys.update(nested)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            nested = self._extract_all_keys(item, prefix=f'{prefix}{key}.')
                            keys.update(nested)
        
        return keys
    
    def validate_all_configs(self):
        """Validate all config files in configs/ directory."""
        config_dir = self.project_root / 'configs'
        
        if not config_dir.exists():
            print(f"❌ Config directory not found: {config_dir}")
            return
        
        results = {}
        for config_file in config_dir.glob('*.json'):
            print(f"\n{'='*80}")
            print(f"Validating: {config_file.name}")
            print('='*80)
            
            result = self.find_zombie_keys(config_file)
            results[config_file.name] = result
            
            print(f"Total keys: {len(result['config_keys'])}")
            print(f"Used keys: {len(result['used_keys'])} ({result['usage_rate']:.1f}%)")
            print(f"Zombie keys: {result['zombie_count']}")
            
            if result['zombie_keys']:
                print("\n⚠️  ZOMBIE KEYS DETECTED:")
                for key in sorted(result['zombie_keys']):
                    print(f"   - {key}")
                print("\nThese keys may be typos or deprecated. Verify they are intentional.")
            else:
                print("\n✅ No zombie keys detected!")
        
        return results
    
    def generate_report(self, results, output_path=None):
        """Generate a markdown report of validation results."""
        if output_path is None:
            output_path = self.project_root / 'results' / 'config_validation_report.md'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Configuration Validation Report\n\n")
            f.write("**Generated**: December 9, 2025\n\n")
            f.write("## Summary\n\n")
            
            total_configs = len(results)
            total_zombie = sum(r['zombie_count'] for r in results.values())
            
            f.write(f"- **Configs analyzed**: {total_configs}\n")
            f.write(f"- **Total zombie keys**: {total_zombie}\n\n")
            
            f.write("## Per-Config Results\n\n")
            
            for config_name, result in results.items():
                f.write(f"### {config_name}\n\n")
                f.write(f"- Total keys: {len(result['config_keys'])}\n")
                f.write(f"- Used keys: {len(result['used_keys'])} ({result['usage_rate']:.1f}%)\n")
                f.write(f"- Zombie keys: {result['zombie_count']}\n\n")
                
                if result['zombie_keys']:
                    f.write("**Unused Keys**:\n")
                    for key in sorted(result['zombie_keys']):
                        f.write(f"- `{key}`\n")
                    f.write("\n")
                else:
                    f.write("OK: All keys are used in code.\n\n")
        
        print(f"\n📄 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate configuration files for zombie keys'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Specific config file to validate (optional)'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate markdown report'
    )
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    
    if args.config:
        # Validate single config
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return 1
        
        result = validator.find_zombie_keys(config_path)
        
        print(f"\nValidation Results for {config_path.name}")
        print('='*80)
        print(f"Total keys: {len(result['config_keys'])}")
        print(f"Used keys: {len(result['used_keys'])} ({result['usage_rate']:.1f}%)")
        print(f"Zombie keys: {result['zombie_count']}")
        
        if result['zombie_keys']:
            print("\n⚠️  ZOMBIE KEYS:")
            for key in sorted(result['zombie_keys']):
                print(f"   - {key}")
        else:
            print("\n✅ No zombie keys!")
        
        return 1 if result['zombie_keys'] else 0
    
    else:
        # Validate all configs
        results = validator.validate_all_configs()
        
        if args.report and results:
            validator.generate_report(results)
        
        # Return error code if any zombies found
        total_zombies = sum(r['zombie_count'] for r in results.values())
        return 1 if total_zombies > 0 else 0


if __name__ == '__main__':
    exit(main())
