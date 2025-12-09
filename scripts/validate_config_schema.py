#!/usr/bin/env python3
"""
JSON Schema Validator for Experiment Configs

Validates all experiment configuration files against the JSON schema
to ensure proper structure and catch typos/zombie keys.

This is BLOCKER-3 fix from the Research Validity Audit.

Usage:
    python scripts/validate_config_schema.py
    
Author: GDSearch Remediation Team
Date: December 9, 2025
"""

import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema not installed. Run: pip install jsonschema")
    sys.exit(1)


def validate_config(config_path, schema_path):
    """Validate single config file against schema.
    
    Args:
        config_path: Path to config JSON file
        schema_path: Path to JSON schema file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(schema_path) as f:
            schema = json.load(f)
    except FileNotFoundError:
        return False, f"Schema file not found: {schema_path}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in schema: {e}"
    
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        return False, f"Config file not found: {config_path}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in config: {e}"
    
    try:
        jsonschema.validate(config, schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, f"{e.message}\nPath: {' -> '.join(str(p) for p in e.path)}"
    except jsonschema.SchemaError as e:
        return False, f"Invalid schema: {e.message}"


def main():
    """Main validation function."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    schema_path = repo_root / 'configs' / 'config_schema.json'
    configs_dir = repo_root / 'configs'
    
    # Config files to validate
    config_files = [
        configs_dir / 'nn_tuning.json',
        configs_dir / 'cifar10_tuning.json',
        configs_dir / 'benchmark_hyperparameters.json'
    ]
    
    print("=" * 80)
    print("JSON SCHEMA VALIDATION REPORT")
    print("=" * 80)
    print(f"Schema: {schema_path.relative_to(repo_root)}")
    print()
    
    all_valid = True
    results = []
    
    for config_file in config_files:
        if not config_file.exists():
            print(f"⚠  {config_file.name}: Not found (skipping)")
            results.append((config_file.name, 'SKIP', 'File not found'))
            continue
        
        valid, error = validate_config(config_file, schema_path)
        if valid:
            print(f"✓  {config_file.name}: VALID")
            results.append((config_file.name, 'PASS', None))
        else:
            print(f"✗  {config_file.name}: INVALID")
            print(f"   Error: {error}")
            print()
            results.append((config_file.name, 'FAIL', error))
            all_valid = False
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, status, _ in results if status == 'PASS')
    failed = sum(1 for _, status, _ in results if status == 'FAIL')
    skipped = sum(1 for _, status, _ in results if status == 'SKIP')
    
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print("=" * 80)
    
    if not all_valid:
        print("\n❌ Config validation FAILED")
        print("\nRecommendations:")
        print("1. Review the errors above")
        print("2. Fix the config files to match the schema")
        print("3. Run this script again to verify fixes")
        sys.exit(1)
    else:
        print("\n✅ All configs are schema-compliant")
        sys.exit(0)


if __name__ == '__main__':
    main()
