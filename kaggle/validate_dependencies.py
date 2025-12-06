#!/usr/bin/env python3
"""
Validate that all dependencies are correctly installed with compatible versions.
Run this after installing requirements to check for conflicts.
"""

import sys
import importlib.metadata
from typing import Dict, List, Tuple

# Define expected version constraints
EXPECTED_VERSIONS = {
    'fsspec': ('==', '2025.3.0'),
    'pyarrow': ('>=', '14.0.0', '<', '20.0.0'),
    'protobuf': ('>=', '3.20.3', '<', '4.0.0'),
    'rich': ('>=', '12.4.4', '<', '14'),
    'click': ('>=', '7.0', '!=', '8.3.0'),
    'cryptography': ('>=', '19.0', '<', '44'),
    'pyOpenSSL': ('>=', '19.1.0', '<=', '24.2.1'),
    'numpy': ('>=', '1.26.0', '<', '2.0'),
}

def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string to tuple of integers."""
    try:
        # Handle versions like '2025.3.0' or '1.26.4'
        return tuple(int(x) for x in version_str.split('.')[:3])
    except:
        return (0, 0, 0)

def check_version(package: str, constraints: tuple) -> Tuple[bool, str]:
    """Check if installed package version meets constraints."""
    try:
        version = importlib.metadata.version(package)
        ver_tuple = parse_version(version)
        
        i = 0
        while i < len(constraints):
            op = constraints[i]
            
            if op == '==':
                expected = constraints[i + 1]
                if version != expected:
                    return False, f"Expected {expected}, got {version}"
                i += 2
            elif op == '>=':
                min_ver = parse_version(constraints[i + 1])
                if ver_tuple < min_ver:
                    return False, f"Version {version} < {constraints[i + 1]}"
                i += 2
            elif op == '<=':
                max_ver = parse_version(constraints[i + 1])
                if ver_tuple > max_ver:
                    return False, f"Version {version} > {constraints[i + 1]}"
                i += 2
            elif op == '<':
                max_ver = parse_version(constraints[i + 1])
                if ver_tuple >= max_ver:
                    return False, f"Version {version} >= {constraints[i + 1]}"
                i += 2
            elif op == '!=':
                bad_ver = constraints[i + 1]
                if version == bad_ver:
                    return False, f"Version {version} is forbidden"
                i += 2
            else:
                i += 1
        
        return True, version
    except importlib.metadata.PackageNotFoundError:
        return False, "NOT INSTALLED"
    except Exception as e:
        return False, f"ERROR: {e}"

def main():
    print("=" * 70)
    print("DEPENDENCY VALIDATION")
    print("=" * 70)
    
    all_ok = True
    
    for package, constraints in EXPECTED_VERSIONS.items():
        ok, message = check_version(package, constraints)
        
        status = "✅" if ok else "❌"
        print(f"{status} {package:20s} {message}")
        
        if not ok:
            all_ok = False
    
    print("=" * 70)
    
    # Check critical imports
    print("\nCRITICAL IMPORTS CHECK:")
    critical_imports = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('transformers', 'HuggingFace Transformers'),
        ('datasets', 'HuggingFace Datasets'),
        ('optuna', 'Optuna'),
        ('mlflow', 'MLflow'),
        ('plotly', 'Plotly'),
    ]
    
    for module_name, display_name in critical_imports:
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {display_name:30s} {version}")
        except ImportError as e:
            print(f"❌ {display_name:30s} NOT INSTALLED")
            all_ok = False
    
    print("=" * 70)
    
    if all_ok:
        print("✅ ALL DEPENDENCIES OK")
        return 0
    else:
        print("❌ SOME DEPENDENCIES HAVE ISSUES - Please reinstall")
        return 1

if __name__ == '__main__':
    sys.exit(main())
