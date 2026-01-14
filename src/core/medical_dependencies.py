"""
Optional Medical Imaging Dependencies Checker and Guidance.

This module provides runtime detection and clear error messages for
optional medical imaging packages (MONAI, medmnist) to help users
understand and resolve missing dependencies.
"""
import logging
from typing import Optional, Dict, Any

# Detection flags
HAS_MEDMNIST = False
HAS_MONAI = False
MEDICAL_IMPORT_ERRORS: Dict[str, str] = {}

# Attempt to import medmnist
try:
    import medmnist  # noqa: F401
    HAS_MEDMNIST = True
except ImportError as e:
    MEDICAL_IMPORT_ERRORS['medmnist'] = str(e)
    logging.debug("medmnist not available: %s", e)

# Attempt to import MONAI
try:
    import monai  # noqa: F401
    HAS_MONAI = True
except ImportError as e:
    MEDICAL_IMPORT_ERRORS['monai'] = str(e)
    logging.debug("MONAI not available: %s", e)


class MedicalDependencyError(ImportError):
    """Raised when required medical imaging dependencies are missing."""
    pass


def require_medmnist(feature_name: str = "Medical imaging experiments") -> None:
    """
    Raise clear error if medmnist is not available.

    Args:
        feature_name: Description of the feature requiring medmnist

    Raises:
        MedicalDependencyError: If medmnist is not installed
    """
    if not HAS_MEDMNIST:
        error_msg = (
            f"\n{'='*70}\n"
            f"MISSING DEPENDENCY: medmnist\n"
            f"{'='*70}\n\n"
            f"{feature_name} requires the 'medmnist' package.\n\n"
            f"To install:\n"
            f"  pip install medmnist\n\n"
            f"Or install all medical dependencies:\n"
            f"  pip install -e .[medical]\n\n"
            f"Documentation: https://medmnist.com/\n"
            f"Original error: {MEDICAL_IMPORT_ERRORS.get('medmnist', 'Unknown')}\n"
            f"{'='*70}\n"
        )
        raise MedicalDependencyError(error_msg)


def require_monai(feature_name: str = "Medical segmentation experiments") -> None:
    """
    Raise clear error if MONAI is not available.

    Args:
        feature_name: Description of the feature requiring MONAI

    Raises:
        MedicalDependencyError: If MONAI is not installed
    """
    if not HAS_MONAI:
        error_msg = (
            f"\n{'='*70}\n"
            f"MISSING DEPENDENCY: MONAI\n"
            f"{'='*70}\n\n"
            f"{feature_name} requires the 'monai' package.\n\n"
            f"To install:\n"
            f"  pip install monai\n\n"
            f"For full functionality including transforms and networks:\n"
            f"  pip install 'monai[all]'\n\n"
            f"Or install all medical dependencies:\n"
            f"  pip install -e .[medical]\n\n"
            f"Documentation: https://docs.monai.io/\n"
            f"Original error: {MEDICAL_IMPORT_ERRORS.get('monai', 'Unknown')}\n"
            f"{'='*70}\n"
        )
        raise MedicalDependencyError(error_msg)


def check_medical_stack(verbose: bool = True) -> Dict[str, bool]:
    """
    Check availability of all medical imaging dependencies.

    Args:
        verbose: If True, print availability report

    Returns:
        Dict mapping package names to availability (True/False)
    """
    availability = {
        'medmnist': HAS_MEDMNIST,
        'monai': HAS_MONAI
    }

    if verbose:
        print("\n" + "="*70)
        print("Medical Imaging Dependencies Status")
        print("="*70)
        for pkg, available in availability.items():
            status = "[OK]" if available else "[MISSING]"
            print(f"  {pkg:20s}: {status}")
            if not available and pkg in MEDICAL_IMPORT_ERRORS:
                print(f"    Error: {MEDICAL_IMPORT_ERRORS[pkg]}")
        print("="*70 + "\n")

        if not all(availability.values()):
            print("To install missing packages:")
            if not HAS_MEDMNIST:
                print("  pip install medmnist")
            if not HAS_MONAI:
                print("  pip install monai")
            print("\nOr install all at once:")
            print("  pip install -e .[medical]\n")

    return availability


def get_install_command(package: str = 'all') -> str:
    """
    Get the install command for medical dependencies.

    Args:
        package: 'medmnist', 'monai', or 'all'

    Returns:
        Installation command string
    """
    if package == 'medmnist':
        return "pip install medmnist"
    elif package == 'monai':
        return "pip install 'monai[all]'"
    else:  # 'all'
        return "pip install -e .[medical]"


def safe_import_medmnist() -> Optional[Any]:
    """
    Safely import medmnist, returning None if unavailable.

    Returns:
        medmnist module if available, None otherwise
    """
    if HAS_MEDMNIST:
        import medmnist
        return medmnist
    return None


def safe_import_monai() -> Optional[Any]:
    """
    Safely import MONAI, returning None if unavailable.

    Returns:
        monai module if available, None otherwise
    """
    if HAS_MONAI:
        import monai
        return monai
    return None


if __name__ == '__main__':
    # When run as script, print dependency status
    check_medical_stack(verbose=True)
