# Reproducibility guidance for PyTorch experiments 🔬✨

This project aims for deterministic behaviour where possible. The following steps reduce nondeterminism.

## Recommended environment / runtime settings
- Set Python-level randomness seeds:
  - set `PYTHONHASHSEED` environment variable (e.g., `export PYTHONHASHSEED=42`)
- In code, early during experiment startup call:

```python
import os
import random
import numpy as np
import torch

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
# PyTorch reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
# Deterministic algorithms (warning: may slow performance)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

- For CUDA deterministic behavior on modern CUDA toolkits, set:
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8` (Linux) or the recommended NVIDIA instructions for your CUDA version

## Code helpers
- We added `src/utils/reproducibility.py` with a helper `enforce_reproducibility(seed=42, deterministic=True)` that performs the above steps.
- Call the helper at the earliest entrypoint for experiments (e.g., in `run_all_kaggle.py` main entry or `src/experiments/run_*` scripts).

## Caveats
- Some operations (e.g., certain CUDA kernels, nondeterministic ops in `torch.ops`) cannot be made deterministic across all hardware/driver combos.
- Deterministic mode may significantly slow down training and disable some optimizations.

## Tests
- Add CI tests that run short `--ultra-quick` seeds and assert deterministic outputs across repeated runs when reproducibility mode is enabled.

For more details, see the PyTorch reproducibility docs relevant to your PyTorch/CUDA version.