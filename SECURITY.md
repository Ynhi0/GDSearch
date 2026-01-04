# Security Guidelines

## Overview
This document outlines security considerations for the GDSearch project, particularly around model checkpoint handling and code execution patterns.

## Model Checkpoint Security

### torch.load() Safety
⚠️ **CRITICAL**: The `torch.load()` function uses Python's `pickle` module, which can execute arbitrary code during deserialization. **Only load checkpoints from trusted sources.**

### Best Practices
1. **Trusted Sources Only**: Only load checkpoints that you created yourself or from verified, trusted sources
2. **Checkpoint Verification**: When sharing checkpoints, provide SHA256 checksums for integrity verification
3. **Sandboxed Loading**: Consider loading untrusted checkpoints in isolated environments (containers, VMs)
4. **Use weights_only Parameter**: When possible, use `torch.load(..., weights_only=True)` (PyTorch 2.6+) to restrict deserialization to tensors only

### Safe Checkpoint Loading Pattern
```python
def torch_load_safe(path: Path, map_location=None, trusted: bool = False):
    """
    Safely load PyTorch checkpoint with explicit trust verification.
    
    Args:
        path: Path to checkpoint file
        map_location: Device mapping for tensors
        trusted: Set to True ONLY if checkpoint is from a known, verified source
        
    Raises:
        ValueError: If checkpoint source is not explicitly marked as trusted
    """
    if not trusted:
        raise ValueError(
            f"Checkpoint {path} not marked as trusted. Only load checkpoints from "
            "verified sources. See SECURITY.md for details."
        )
    
    try:
        # PyTorch 2.6+ supports weights_only for additional safety
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions
        print(f"⚠️  WARNING: Loading {path} without weights_only restriction")
        return torch.load(path, map_location=map_location)
```

### Checkpoint Integrity Verification
```bash
# Generate checksum for distribution
sha256sum checkpoint.pt > checkpoint.pt.sha256

# Verify before loading
sha256sum -c checkpoint.pt.sha256
```

## Code Execution Patterns

### Prohibited Patterns
❌ **NEVER USE** in production code:
- `exec()` or `eval()` with user-provided input
- `subprocess.run(..., shell=True)` with user-controlled arguments
- `os.system()` with user-controlled strings
- `pickle.load()` on untrusted data sources

### Safe Alternatives
✅ **USE INSTEAD**:
- `importlib.util.spec_from_file_location()` for dynamic imports
- `subprocess.run([...])` with list-form arguments (no shell=True)
- `ast.literal_eval()` for safe string-to-Python conversion
- JSON or other non-executable serialization formats

### Example: Safe Module Loading
```python
import importlib.util

def load_module_safely(file_path: Path, module_name: str):
    """Safely load a Python module from file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # Controlled execution
        return module
    raise ImportError(f"Cannot load module from {file_path}")
```

### Example: Safe Subprocess Execution
```python
import subprocess

# ❌ UNSAFE (shell injection risk)
subprocess.run(f"python {user_script}", shell=True)

# ✅ SAFE (list-form arguments)
subprocess.run([sys.executable, user_script], check=True)
```

## Dependencies

### Pinned Versions
All production dependencies are pinned to specific versions in `requirements.txt` to prevent supply chain attacks.

```
torch==2.6.0
numpy==1.26.4
transformers==4.47.1
```

### Updating Dependencies
1. Test updates in isolated environment first
2. Review changelogs for security fixes
3. Update pinned versions in requirements.txt
4. Run full test suite before committing

## Reporting Security Issues

If you discover a security vulnerability in GDSearch:
1. **DO NOT** open a public GitHub issue
2. Email the maintainers directly with details
3. Include: affected versions, exploit scenario, proposed fix
4. Allow 90 days for patch development before public disclosure

## Security Audit History

| Date       | Auditor           | Findings                          | Status   |
|------------|-------------------|-----------------------------------|----------|
| 2024-01-XX | Internal Review   | exec() usage in tests             | FIXED    |
| 2024-01-XX | Internal Review   | shell=True in subprocess calls    | FIXED    |
| 2024-01-XX | Internal Review   | Broad exception handlers          | IN PROGRESS |

## References
- [PyTorch Security](https://pytorch.org/docs/stable/notes/serialization.html#security)
- [Python Pickle Security](https://docs.python.org/3/library/pickle.html#restricting-globals)
- [OWASP Code Injection](https://owasp.org/www-community/attacks/Code_Injection)
- [Bandit Security Linter](https://bandit.readthedocs.io/)

---
**Last Updated**: 2024-01-XX  
**Version**: 1.0
