#!/usr/bin/env python3

"""
GDSearch Dependency Installer
Installs all required dependencies with progress tracking
"""

import subprocess
import sys
import os

def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def install_dependencies():
    """Install all dependencies from requirements.txt"""
    print_header("GDSEARCH DEPENDENCY INSTALLER")
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"ERROR: {requirements_file} not found!")
        return False
    
    print(f"Installing dependencies from {requirements_file}...")
    print("This may take several minutes...\n")
    
    try:
        # Install dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        
        print("\n" + "="*70)
        print("SUCCESS: All dependencies installed!".center(70))
        print("="*70 + "\n")
        
        # Verify key imports
        print("Verifying installations...")
        try:
            import torch
            print(f"  OK torch {torch.__version__}")
        except ImportError:
            print("  X torch failed to import")
            
        try:
            import pandas as pd
            print(f"  OK pandas {pd.__version__}")
        except ImportError:
            print("  X pandas failed to import")
            
        try:
            import numpy as np
            print(f"  OK numpy {np.__version__}")
        except ImportError:
            print("  X numpy failed to import")
            
        try:
            import scipy
            print(f"  OK scipy {scipy.__version__}")
        except ImportError:
            print("  X scipy failed to import")
        
        print("\nRun 'python scripts/quick_validation_test.py' to verify the installation.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Installation failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        return False

if __name__ == "__main__":
    success = install_dependencies()
    sys.exit(0 if success else 1)
