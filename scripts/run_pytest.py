import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
import pytest

if __name__ == '__main__':
    # Run pytest with quiet mode and no warnings
    # Try running pytest without capture to avoid I/O-on-closed-file errors
    rc = pytest.main(['-q', '-s', '--disable-warnings'])
    sys.exit(rc)
