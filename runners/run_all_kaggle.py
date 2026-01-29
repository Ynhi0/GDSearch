"""Compatibility shim to expose top-level run_all_kaggle modules as runners.run_all_kaggle

This file keeps older import paths working for scripts/tests that import
from runners.run_all_kaggle import ...
"""
# Re-export everything from the top-level module
from run_all_kaggle import *  # noqa: F401,F403
