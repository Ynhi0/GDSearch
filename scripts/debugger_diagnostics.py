"""Diagnostics to help with debugger issues (frozen modules, missed breakpoints).

Run locally when you observe missing breakpoints or frozen-module warnings in logs.

Example:
    python scripts/debugger_diagnostics.py
"""
from __future__ import annotations
import os
import sys
import platform
import logging

logging.basicConfig(level=logging.INFO)


def main():
    print("Debugger diagnostics")
    print("Python:", sys.version)
    print("Platform:", platform.platform())

    # Print relevant environment variables
    for k in ["PYDEVD_USE_FRAME_EVAL", "PYDEVD_DEBUG", "PYDEVD_LOG_DIR", "PYTHONHASHSEED"]:
        print(f"{k}={os.environ.get(k)}")

    try:
        import debugpy
        print("debugpy version:", getattr(debugpy, '__version__', 'unknown'))
    except Exception as e:
        print("debugpy not installed:", e)

    try:
        import pydevd
        print("pydevd version:", getattr(pydevd, '__version__', 'unknown'))
    except Exception as e:
        print("pydevd not installed:", e)

    print("\nAdvice:\n - If you see frozen module messages, set 'justMyCode': false in VS Code launch.json.\n - Try setting PYDEVD_USE_FRAME_EVAL=NO if breakpoints are missed in CPython 3.11+ with certain debugpy builds.")


if __name__ == '__main__':
    main()